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

    def fit_all(self, vers_dics):
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
    def fit(self,X,y):
        # find intersection of ALL changes
        X_sets = []
        for changes in X:
            X_sets.append(set(changes))
        #s.intersection_update(t)
        intersection = set(X_sets[0])
        for idx in range(1, len(X_sets)):
            #intersection.insersection_update(all_changes[idx])
            intersection &= X_sets[idx]

        # remove intersection from all sets
        for l in X_sets:
            l -= intersection

        # now separate by label
        # find number of unique labels
        y_set = set(y)
        # convert the set to the list
        unique_y = (list(y_set))

        num_versions = len(unique_y)

        # create a dictionary that will have the versions as the keys, rules as values
        changes_dic = {}
        rules_dic = {}
        for lab in unique_y:
            changes_dic[lab] = []
            rules_dic[lab] = []

        # separate changes by label
        for label, changes in zip(y, X):
            #print(label)
            changes_dic[label].append(changes)

        #find intersections within list
        intersections = {}

        for lab in unique_y:
            cur_int = set(changes_dic[lab][0])
            for i in range(1, len(changes_dic[lab])):
                cur_int &= set(changes_dic[lab][i])
            intersections[lab] = list(cur_int)

        # create rules
        for lab in unique_y:
            cur_int = intersections[lab]
            cur_other_lab = unique_y
            cur_other_lab.remove(lab)
            #print(indices)
            for change in cur_int:
                unique_rule = True
                for y in cur_other_lab:
                    if change in intersections[y]:
                        unique_rule = False
                        break
                if unique_rule:
                    rules_dic[lab].append(change)
        return rules_dic

    def predict_all(self, sep_test_dics):
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
            print(label, rel_rules)
            preds_dic[label] = self.predict(list_dics[label], rel_rules)
        return preds_dic

    def predict(self, X, rel_rules):
        # create a list of predictions for the Xs
        #print("Number of changesets: ", len(X))
        rule_keys = rel_rules.keys();
        preds = []
        for changes in X:
            given_lab = False
            for key in rule_keys:
                if len(rel_rules[key]) != 0 and set(rel_rules[key]) <= set(changes):
                    preds.append(key)
                    given_lab = True
                    break
            if not given_lab:
                preds.append("???")
        print(preds)
        #input("Enter to continue...")
        return preds
