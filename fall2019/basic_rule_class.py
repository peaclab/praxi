from collections import OrderedDict
from orderedset import OrderedSet
from itertools import combinations
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
        # takes a list of dictionaries w/ application, label, and changes
        self.rules = {}
        list_dics = {}
        all_labels = vers_dics.keys()

        # Need a new dictionary separated by application
        # for each application, have list of changes and versions
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
        self.rules = {}
        labels = list_dics.keys()
        for lab in labels:
            self.rules[lab] = self.fit(list_dics[lab]['X'], list_dics[lab]['y'])
        with open('hyb_rules.yaml', 'w') as outfile:
            yaml.dump(self.rules, outfile, default_flow_style=False)

    #def fit(self, X, y): # X is list of lists, little lists have tags, y are VERSIONS
    def fit(self,X,y,max_num_rules=1):
        # takes: list of changes, corresponding list of labels
        # find intersection of ALL changes and remove?
        labels = list(set(y))
        partitioned_X = [ [] for i in range(len(labels)) ]
        unions = []

        for cur_changes, cur_label in zip(X,y):
            idx = labels.index(cur_label)
            partitioned_X[idx].append(cur_changes)

        for partition in partitioned_X:
            cur_union = set(partition[0])
            for cur_changes in partition:
                cur_union |= set(cur_changes)
            unions.append(list(cur_union))

        overlap = get_overlap(unions)
        new_partitioned_X = [ [] for i in range(len(labels)) ]
        for cur_changes, cur_label in zip(X,y):
            idx = labels.index(cur_label)
            changes_no_overlap = [x for x in cur_changes if x not in overlap]
            new_partitioned_X[idx].append(changes_no_overlap)

        rules = {}
        for changesets, label in zip(new_partitioned_X, labels):
            label_intersection = set(changesets[0])
            for c in changesets:
                label_intersection &= set(c)
            label_intersection = list(label_intersection)
            if len(label_intersection)==0:
                rules[label] = "???"
            else:
                rules[label] = label_intersection[0]
        return rules

    def predict_all(self, sep_test_dics): # prediction wrapper
        print("Starting predictions")
        test_labels = sep_test_dics.keys()

        preds_dic = {}
        list_dics = {}
        for label in test_labels:
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
        preds = []
        for changes in X:
            given_lab = False
            for key in rule_keys: # relrules[key] will have only one element
                if len(rel_rules[key]) != 0 and rel_rules[key] in changes:
                    preds.append(key)
                    given_lab = True
                    break
            if not given_lab:
                preds.append("???")
        #print("Predictions: ", preds)
        #input("Enter to continue...")
        return preds


def get_overlap(unions):
    combos = list(combinations(unions, 2))
    intersections = []
    for combo in combos:
        intersections.append(list(set(combo[0]) & set(combo[1])))
    overlap = set(intersections[0])
    for i in intersections:
        overlap |= set(i)
    overlap = list(overlap)
    return overlap
