""" Pseudocode for rule generation """


from collections import OrderedDict
from orderedset import OrderedSet

class RuleBasedTags:
    """Scikit wrapper"""
    def __init__(self, threshold=0.5, num_rules=1, max_index=20,
                 string_rules=False, unknown_label='???'):
        self.threshold = threshold
        self.max_index = max_index
        self.string_rules = string_rules
        self.unknown_label = unknown_label
        self.num_rules = num_rules

    def fit(self, X, y): # X is list of lists, little lists have tags, y are labels
        label_to_tokens = self.transform_tagsets(X, y)
        # get inverse map
        token_to_labels = get_token_to_labels(label_to_tokens)
        # get map from labels to categorized tokens
        label_to_token_groups = get_label_to_token_groups(token_to_labels)
        # find duplicates and filter them out
        duplicates = get_duplicates(label_to_tokens, token_to_labels, label_to_token_groups)
        for k in label_to_tokens:
            if k in duplicates:
                del label_to_tokens[k]
        # get inverse map again
        token_to_labels = get_token_to_labels(label_to_tokens)
        # get map from pabels to categorized items AGAIN
        label_to_token_groups = get_label_to_token_groups(token_to_labels)
        # Generate rules!
        rules = get_rules(label_to_tokens, token_to_labels, label_to_token_groups,
                          limit=self.num_rules, max_index=self.max_index,
                          string_rules=self.string_rules)
        self.rules=OrderedDict()
        for k, v in rules.items():
            if k in y:
                self.rules[k] = v
        print("Finished rule generation")

    def predict(self, X, ntags=None):
        if ntags is None:
            unravel_result = True
            ntags = [1 for _ in range(len(X))]
        else:
            unravel_result = False
        predictions = []
        for n_preds, tags in zip(ntags, X):
            cur_predictions = {}
            for label_tested, label_rules in self.rules.items():
                n_rules_satisfied = 0
                n_rules = len(label_rules)
                if n_rules == 0:
                    print(label_tested, " has no rules")
                    continue
                for rule in label_rules:
                    rule_satisfied = True
                    for triplet in rule:
                        token = triplet[0]
                        inside = (triplet[1] != 'outside vs')
                        if inside == (len([v for v in tags
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
        return predictions

    def transform_tagsets(self, tagsets, labels, take_max=False):  # Changesets as dictionaries
        res = OrderedDict()
        for data, label in zip(tagsets, labels):
            for token in data:
                if label not in res:
                    res[label] = OrderedDict()
                if token not in res[label]:
                    res[label][token] = 1
                else:
                    res[label][token] += 1
        newres = dict()
        for label in res:
            newres[label] = set()
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
    """ Returns inverse map: from tokens to sets of labels """
    token_to_labels = OrderedDict()
    for label in label_to_tokens:
        for token in label_to_tokens[label]:
            if token not in token_to_labels:
                token_to_labels[token] = OrderedSet()
            token_to_labels[token].add(label)
    return token_to_labels

def get_label_to_token_groups(token_to_labels):
    """ Returns categorized corpus, dictionary from labels
        to groups of tokens, indexed with natural numbers; index
        of a group shows in how many labels each token from the
        group is present
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
    print(label_to_token_groups['mgetty-voice'])
    input("Enter to continue...")
    return label_to_token_groups

def get_duplicates(label_to_tokens, token_to_labels, label_to_token_groups):
    """ Returns labels, not all, that have sets of tokens identical to other
        labels. From each group of identical labels one label goes to representatives
        all other labels from each group go to <duplicates>"""

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
                print("Duplicates:". label, other_label)
        return duplicates

def get_rules(label_to_tokens, token_to_labels, label_to_token_groups,
              limit=1, max_index=5, string_rules=False):
    """ Generates a dictionary from labels to sets of rules """
    rules = OrderedDict()
    for label in label_to_token_groups:
        rules[label] = get_rules_per_label(label, label_to_tokens, token_to_labels,
                                           label_to_token_groups, limit, max_index, string_rules)
    return rules

def get_rules_per_label(label, label_to_tokens, token_to_labels, label_to_token_groups,
                        limit=1, max_index=0, string_rules=False):
    """ Generate <limit> rules for a labels

        Each rule is a list of <index> triplets
        Each rule contains one triplet of the format:
        (<token>, 'unique to', <index>)
        Means that the token appears in index different labels including <label>
        All labels listed once in other triplets as other_label
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
                rules.append(token + ' unique to ' + str(index))
            else:
                rule.append((token, ' unique to ', str(index)))
            for other_label in token_to_labels[token]:
                if other_label == label:
                    continue
                plus_diff = label_to_tokens[label] - \
                    label_to_tokens[other_label]
                minus_diff = label_to_tokens[other_label] - \
                    label_to_tokens[label]
                assert(len(plus_diff)+len(minus_diff)) > 0
                plus_diff -= used_tokens
                minus_diff -= used_tokens
                if len(plus_diff) > 0:
                    if string_rules:
                        rule.append(list(plus_diff)[0] + ' inside vs ' + other_label)
                    else:
                        rule.append((list(plus_diff)[0], ' inside vs ', other_label))
                elif len(minus_diff) > 0:
                    if string_rules:
                        rule.append(list(minus_diff)[0] + ' inside vs ' + other_label)
                    else:
                        rule.append((list(minus_diff)[0], ' inside vs ', other_label))
                else:
                    break
            if len(rule) < index:
                continue
            rules.append(rule)
            for triplet in rule:
                used_tokens.add(triplet[0])
            if len(rules) >= limit:
                return rules
    print(rules)
    return rules
