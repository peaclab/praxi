from collections import OrderedDict
from orderedset import OrderedSet
import yaml

class RuleBasedTags:
    """Scikit wrapper"""
    def __init__(self, threshold=0.5, num_rules=1, max_index=20,
                 string_rules=False, unknown_label='???'):
        self.threshold = threshold
        self.max_index = max_index
        self.string_rules = string_rules
        self.unknown_label = unknown_label
        self.num_rules = num_rules
        self.total_rules = 0;
        self.total_apps = 0;
        self.total_versions = 0;

    def fit_all(self, vers_dics): # fit wrapper
        self.rules = {}
        list_dics = {}
        all_labels = vers_dics.keys()
        #print(all_labels)
        #input("Press enter to continue...")
        for label in all_labels:
            if label not in list_dics:
                list_dics[label] = {}
                list_dics[label]['X'] = []
                list_dics[label]['y'] = []
                self.total_apps += 1
            rel_dics = vers_dics[label]
            for dic in rel_dics:
                changelist = dic['changes']
                list_dics[label]['X'].append(changelist)
                version = dic['version']
                list_dics[label]['y'].append(version)
        #self.rules = {}
        labels = list_dics.keys()
        for lab in labels:
            #print(lab)
            self.rules[lab] = self.fit(list_dics[lab]['X'], list_dics[lab]['y'])
        with open('hyb_rules.yaml', 'w') as outfile:
            yaml.dump(self.rules, outfile, default_flow_style=False)

    #def fit(self, X, y): # X is list of lists, little lists have tags, y are VERSIONS
    def fit(self,X,y,max_num_rules=1):
        # find intersection of ALL changes
        """inter = set(X[0])
        for idx in range(1,len(X)):
            inter &= set(X[idx])
        inter = list(inter)
        if len(inter) != 0:
            newX = []
            for changes in X:
                 new_changes = [x for x in changes if x not in inter]
                 newX.append(new_changes)
            X = newX"""

        label_to_tokens = self.transform_tagsets(X, y)
        labels = label_to_tokens.keys()
        self.total_versions += len(labels)

        #self.rules = {}
        #for l in labels:
        #    rules[l]=[]
        token_to_labels = self.get_token_to_labels(label_to_tokens)
        label_to_token_groups = get_label_to_token_groups(token_to_labels)

        rules, numrules = get_rules(label_to_tokens, token_to_labels, label_to_token_groups,
                               limit=1, max_index=5, string_rules=False)
        self.total_rules += numrules

        """for token in token_to_labels.keys():
            if len(token_to_labels[token]) == 1:
                if token_to_labels[token][0] not in rules: # only take one rule
                    self.rules[token_to_labels[token][0]] = token
                    self.total_rules += 1"""
        #print("Rules:", rules)
        #input("Enter to continue");
        #print(rules)
        return rules

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
                if res[label][token] < maxval: # must appear in every changeset
                    break
                newres[label].add(token)
        return newres

    def get_token_to_labels(self, label_to_tokens):
        """ Returns inverse map: from tokens to sets of labels """
        token_to_labels = OrderedDict()
        for label in label_to_tokens:
            for token in label_to_tokens[label]:
                if token not in token_to_labels:
                    token_to_labels[token] = OrderedSet()
                token_to_labels[token].add(label)
        return token_to_labels

    def predict_all(self, sep_test_dics): # prediction wrapper
        print("Starting predictions")
        test_labels = sep_test_dics.keys()

        preds_dic = {}
        list_dics = {}
        for label in test_labels:
            #print(label)
            if label not in list_dics:
                list_dics[label] = []
                rel_dics = sep_test_dics[label]
                for dic in rel_dics:
                    changes = dic['changes']
                    list_dics[label].append(changes)
        for label in list_dics.keys():
            rel_rules = self.rules[label]
            #print(label, rel_rules)
            preds_dic[label] = self.predict(list_dics[label], rel_rules)
        return preds_dic

    def predict(self, X, rel_rules):
        # create a list of predictions for the Xs
        #print("Number of changesets: ", len(X))
        rule_keys = rel_rules.keys(); # only want rules for given app
        predictions = []
        for changes in X:
            cur_predictions = {}
            given_lab = False
            for label_tested, label_rules in rel_rules.items():
                #print(label_tested, label_rules)
                #input("enter to continue...")
                n_rules_satisfied = 0
                n_rules = len(label_rules)
                if n_rules == 0:
                    #print("%s has no rules", label_tested)
                    #preds.append("???")
                    continue
                for rule in label_rules:
                    rule_satisfied = True
                    for trip in rule:
                        token = trip[0]
                        inside = (trip[1] != 'outside vs')
                        if inside == (len([v for v in changes if token== v[-len(token):]]) == 0):
                            rule_satisfied = False
                            break
                    if rule_satisfied:
                        n_rules_satisfied += 1
                cur_predictions[label_tested] = n_rules_satisfied/n_rules
            result = []
            for _ in range(1): # always making one prediction
                if (not cur_predictions) or (max(cur_predictions.values())==0) or (
                        max(cur_predictions.values()) < self.threshold):
                    result.append(self.unknown_label)
                else:
                    best_pred = max(cur_predictions, key=cur_predictions.get)
                    #print(best_pred)
                    result.append(best_pred)
                    del cur_predictions[best_pred]
            predictions.append(result[0])
        return predictions

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


def get_rules(label_to_tokens, token_to_labels, label_to_token_groups,
              limit=1, max_index=5,
              string_rules=False):
    """
    Generates a dictionary from labels to sets of rules.
    See description of <get_rules_per_label> for more details.
    """
    rules = OrderedDict()
    numrules = 0
    for label in label_to_token_groups:
        rules[label] = get_rules_per_label(
            label, label_to_tokens, token_to_labels,
            label_to_token_groups, limit, max_index, string_rules)
        if len(rules[label]) != 0:
            numrules += 1
    return rules, numrules

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
    # GENERATING RULES FOR A SINGLE LABEL!!!
    assert (label in label_to_token_groups)
    #print(label_to_token_groups)
    #input("Enter to continue")
    rules = []
    used_tokens = OrderedSet()
    for index in sorted(label_to_token_groups[label].keys()): # with my dataset, index will be 3 at most
        if index > max_index and max_index > 0:
            break
        for token in label_to_token_groups[label][index]:
            if token in used_tokens: # token already used in...
                continue
            rule = []
            if string_rules:
                rule.append(token + ' unique to ' + str(index))
            else:
                rule.append((token, 'unique to', str(index)))
            for other_label in token_to_labels[token]: # if another label has the given token
                if other_label == label:
                    continue
                # in label but not other label
                plus_diff = label_to_tokens[label] - \
                    label_to_tokens[other_label]
                # in other label but not label
                minus_diff = label_to_tokens[other_label] - \
                    label_to_tokens[label]
                # if either is not empty (they should never both be empty)
                if ((len(plus_diff) + len(minus_diff)) == 0):
                    # identical labels
                    #print("Identical labels")
                    continue
                plus_diff -= used_tokens # get rid of toens that have already been used
                minus_diff -= used_tokens
                if len(plus_diff) > 0:
                    if string_rules:
                        rule.append(
                            list(plus_diff)[0] + ' inside vs ' + other_label) # only need 1
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
            for trip in rule:
                used_tokens.add(trip[0])
            if len(rules) >= limit:
                return rules
    return rules
