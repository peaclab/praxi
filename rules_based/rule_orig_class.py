#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from orderedset import OrderedSet
import logging


class RuleBased:
    """ scikit-style wrapper """
    def __init__(self, threshold=0.5, num_rules=1, max_index=20, string_rules=False,
                 unknown_label='???', filter_method='vlad'):
        self.threshold = threshold
        self.max_index = max_index
        self.string_rules = string_rules
        self.unknown_label = unknown_label
        self.filter_method = filter_method
        self.num_rules = num_rules

    def fit(self, X, y, csids=None):
        # X is list of changesets, y is list of labels
        X, y = self._filter_multilabels(X, y)
        if self.filter_method == 'vlad':
            label_to_tokens = self._transform_anthony_intersection(X, y)
        elif self.filter_method == 'intersect':
            label_to_tokens = self._intersect_packages(X, y)
        elif self.filter_method == 'take_max':
            label_to_tokens = self._transform_anthony_intersection(X, y, take_max=True)
        else:
            raise ValueError("Unknown filter method %s" % self.filter_method)
        # # Filter out labels given by yum that refer to i686 architecture
        # label_to_tokens = {k: v for k, v in label_to_tokens.items()
        #                    if k[-5:] != '.i686'}
        # Get the inverse map
        token_to_labels = get_token_to_labels(label_to_tokens)
        # Get the map from labels to categorized tokens
        label_to_token_groups = get_label_to_token_groups(token_to_labels)
        # Find duplicates
        duplicates = get_duplicates(label_to_tokens, token_to_labels,
                                    label_to_token_groups)
        # Filter out duplicates from the corpus
        for k in label_to_tokens:
            if k in duplicates:
                del label_to_tokens[k]
        # Again get the inverse map
        token_to_labels = get_token_to_labels(label_to_tokens)
        # Again get the map from labels to categorized tokens
        label_to_token_groups = get_label_to_token_groups(token_to_labels)
        # Generate rules for all labels
        rules = get_rules(label_to_tokens, token_to_labels,
                          label_to_token_groups, limit=self.num_rules,
                          max_index=self.max_index,
                          string_rules=self.string_rules)
        # Filter out rules for labels that are not in Anthony's data
        self.rules = OrderedDict()
        for k, v in rules.items():
            if k in y:
                self.rules[k] = v
        logging.info('Finished rule generation')

    def predict(self, X, csids=None, ntags=None):
        if ntags is None:
            unravel_result = True
            ntags = [1 for _ in range(len(X))]
        else:
            unravel_result = False
        predictions = []
        for n_preds, changes in zip(ntags, X):
            cur_predictions = {}
            for label_tested, label_rules in self.rules.items():
                n_rules_satisfied = 0
                n_rules = len(label_rules)
                if n_rules == 0:
                    logging.info("%s has no rules", label_tested)
                    continue
                for rule in label_rules:
                    rule_satisfied = True
                    for triplet in rule:
                        token = triplet[0]
                        inside = (triplet[1] != 'outside vs')
                        if inside == (len([v for v in changes
                                           if token == v[-len(token):]]) == 0):
                            rule_satisfied = False
                            break
                    if rule_satisfied:
                        n_rules_satisfied += 1
                cur_predictions[label_tested] = n_rules_satisfied / n_rules
            result = []
            for _ in range(n_preds):
                if (not cur_predictions) or (
                        max(cur_predictions.values()) == 0) or (
                        max(cur_predictions.values()) < self.threshold and
                        n_preds == 1):
                    result.append(self.unknown_label)
                else:
                    best_pred = max(cur_predictions, key=cur_predictions.get)
                    result.append(best_pred)
                    del cur_predictions[best_pred]
            if unravel_result:
                predictions.append(result[0])
            else:
                predictions.append(result)
        logging.info('Finished rule checking')
        return predictions

    def top_k_tags(self, X, ntags):
        return self.predict(X, ntags=ntags)

    def get_args(self):
        return 'threshold: {}, max_index: {}, num_rules: {}'.format(
            self.threshold, self.max_index, self.num_rules)

    def _filter_multilabels(self, X, y):
        new_X = []
        new_y = []
        for data, labels in zip(X, y):
            if isinstance(labels, list) and len(labels) == 1:
                new_X.append(data)
                new_y.append(labels[0])
            elif isinstance(labels, str):
                new_X.append(data)
                new_y.append(labels)
        return new_X, new_y

    def _intersect_packages(self, changesets, labels):
        """Keep only files present in every instance of a package."""
        res = OrderedDict()
        for data, label in zip(changesets, labels):
            if label in res:
                res[label] &= OrderedSet(data)
                if not len(res[label]):
                    logging.warning("No common files for package %s" % label)
            else:
                res[label] = OrderedSet(data)
        return res

    def _transform_anthony_intersection(self, changesets, labels, take_max=False):
        res = OrderedDict()
        # res[package_name][file_name] = no. of occurances
        for data, label in zip(changesets, labels):
            for token in data:
                if label not in res:
                    res[label] = OrderedDict()
                if token not in res[label]:
                    res[label][token] = 1
                else:
                    res[label][token] += 1
        newres = OrderedDict()
        # newres[package_name] = set(file_names) s.t.
        # freq. of file satisfies mystery_vlad_condition
        for label in res:
            newres[label] = OrderedSet()
            maxval = max(res[label].values())
            for token in sorted(res[label], key=res[label].get, reverse=True):
                mystery_vlad_condition = (
                    (res[label][token] != maxval
                        and len(newres[label]) > 50) or
                    (res[label][token] < 0.94 * maxval
                        and len(newres[label]) >= 40) or
                    (res[label][token] < 0.88 * maxval
                        and len(newres[label]) >= 26) or
                    (res[label][token] < 0.8 * maxval
                        and len(newres[label]) >= 16) or
                    (res[label][token] < 0.7 * maxval
                        and len(newres[label]) >= 10) or
                    (res[label][token] < 0.6 * maxval
                        and len(newres[label]) >= 8) or
                    (res[label][token] < 0.5 * maxval
                        and len(newres[label]) >= 6)
                )
                if take_max:
                    if res[label][token] != maxval:
                        break
                else:
                    if mystery_vlad_condition:
                        break
                newres[label].add(token)
        return newres


def get_token_to_labels(label_to_tokens):
    """
    Returns the inverse map: a dictionary from tokens to sets of labels.
    """
    token_to_labels = OrderedDict()
    for label in label_to_tokens:
        for token in label_to_tokens[label]:
            if token not in token_to_labels:
                token_to_labels[token] = OrderedSet()
            token_to_labels[token].add(label)
    return token_to_labels


def get_label_to_token_groups(token_to_labels):
    """
    Returns a categorized corpus. It's a dictionary from labels to groups
    of tokens. These groups are indexed with natural numbers. Index of a
    group shows in how many labels each token from this group is present.
    """
    label_to_token_groups = OrderedDict()
    for token in token_to_labels:
        for label in token_to_labels[token]:
            index = len(token_to_labels[token])
            if label not in label_to_token_groups:
                label_to_token_groups[label] = OrderedDict()
            if index not in label_to_token_groups[label]:
                label_to_token_groups[label][index] = OrderedSet()
            label_to_token_groups[label][index].add(token)
    return label_to_token_groups


def get_duplicates(label_to_tokens, token_to_labels, label_to_token_groups):
    """
    Returns labels, not all, that have sets of tokens identical to other
    labels. From each group of identical labels one label goes to
    representatives. All the other labels from each group go to <duplicates>.
    """
    duplicates = OrderedSet()
    for label in sorted(label_to_tokens.keys()):
        if label in duplicates:
            continue
        first_index = sorted(label_to_token_groups[label].keys())[0]
        first_token = list(label_to_token_groups[label][first_index])[0]
        potential_duplicates = token_to_labels[first_token]
        for other_label in sorted(list(potential_duplicates)):
            if (other_label <= label) or (other_label in duplicates):
                continue
            if label_to_tokens[label] == label_to_tokens[other_label]:
                duplicates.add(other_label)
                logging.info(
                    'Duplicates: {0} = {1}'.format(label, other_label))
    return duplicates


def get_rules_per_label(label, label_to_tokens, token_to_labels,
                        label_to_token_groups, limit=1, max_index=0,
                        string_rules=False):
    """
    Generates rules, at most <limit>, for a specified <label>.
    Each rule is a list of <index> triplets.
    Each rules includes exactly one triplet
    of the format:
        (*) (<token>, 'unique to', <index>)
    It means that <token> appears exactly in <index> different labels including
    <label>. All these labels, except <label>, are listed exactly once in other
    triplets as <other_label>. A triplet of this format always goes first.
    Other triplets have the formats:
        (1) (<token>, 'inside vs', <other_label>) or
        (2) (<token>, 'outside vs', <other_label>)
    It means that <token> distinguishes <label> from <other_label>. Format (1)
    means that <token> is in <label> but not in <other_label>. Format (2) means
    that <token> is in <other_label> but not in <label>.
    Rules are ordered in a list according to <index>. There could be less rules
    than <limit>. Across all rules each token can appear only once in a triplet
    of format (*) and only once in a triplet of format (1) or (2). This
    guarantees that changes to one token will affect at most two rules. It's
    also guaranteed that rules have the smallest possible <indeces> under the
    requirement given above.
    """
    assert (label in label_to_token_groups)
    rules = []
    used_tokens = OrderedSet()
    for index in sorted(label_to_token_groups[label].keys()):
        if index > max_index and max_index > 0:
            break
        for token in label_to_token_groups[label][index]:
            if token in used_tokens:
                continue
            rule = []
            if string_rules:
                rule.append(token + ' unique to ' + str(index))
            else:
                rule.append((token, 'unique to', str(index)))
            for other_label in token_to_labels[token]:
                if other_label == label:
                    continue
                plus_diff = label_to_tokens[label] - \
                    label_to_tokens[other_label]
                minus_diff = label_to_tokens[other_label] - \
                    label_to_tokens[label]
                assert (len(plus_diff) + len(minus_diff)) > 0
                plus_diff -= used_tokens
                minus_diff -= used_tokens
                if len(plus_diff) > 0:
                    if string_rules:
                        rule.append(
                            list(plus_diff)[0] + ' inside vs ' + other_label)
                    else:
                        rule.append(
                            (list(plus_diff)[0], 'inside vs', other_label))
                elif len(minus_diff) > 0:
                    if string_rules:
                        rule.append(
                            list(minus_diff)[0] + ' outside vs ' + other_label)
                    else:
                        rule.append(
                            (list(minus_diff)[0], 'outside vs', other_label))
                else:
                    break
            if len(rule) < index:
                continue
            rules.append(rule)
            for triplet in rule:
                used_tokens.add(triplet[0])
            if len(rules) >= limit:
                return rules
    return rules


def get_rules(label_to_tokens, token_to_labels, label_to_token_groups,
              limit=1, max_index=5, string_rules=False):
    """
    Generates a dictionary from labels to sets of rules.
    See description of <get_rules_per_label> for more details.
    """
    rules = OrderedDict()
    for label in label_to_token_groups:
        rules[label] = get_rules_per_label(
            label, label_to_tokens, token_to_labels,
            label_to_token_groups, limit, max_index, string_rules)
    return rules
