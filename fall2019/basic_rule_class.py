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
        self.rules = {}
        labels = list_dics.keys()
        for lab in labels:
            self.rules[lab] = self.fit(list_dics[lab]['X'], list_dics[lab]['y'])
        with open('hyb_rules.yaml', 'w') as outfile:
            yaml.dump(self.rules, outfile, default_flow_style=False)

    #def fit(self, X, y): # X is list of lists, little lists have tags, y are VERSIONS
    def fit(self,X,y,max_num_rules=1):
        # find intersection of ALL changes
        label_to_tokens = self.transform_tagsets(X, y)
        labels = label_to_tokens.keys()
        self.total_versions += len(labels)

        rules = {}
        #for l in labels:
        #    rules[l]=[]
        token_to_labels = self.get_token_to_labels(label_to_tokens)
        for token in token_to_labels.keys():
            if len(token_to_labels[token]) == 1:
                #print(token_to_labels[token])
                if token_to_labels[token][0] not in rules: # only take one rule
                    rules[token_to_labels[token][0]] = token
                    self.total_rules += 1

        rule_tokens = rules.keys()
        for label in labels:
            if label not in rule_tokens:
                rules[label] = "???"
        #print("Rules:", rules)
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
                if res[label][token] < (maxval):
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
