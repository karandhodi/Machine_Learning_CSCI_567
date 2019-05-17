import numpy as np
import utils as Util

"""import data
from sklearn.metrics import accuracy_score

features, labels = data.sample_decision_tree_data()

dTree = DecisionTree()
dTree.train(features, labels)

X_test, y_test = data.sample_decision_tree_test()

y_est_test = dTree.predict(X_test)"""


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    # TODO: train Decision Tree
    def train(self, features, labels):
        
        assert (len(features) > 0)
        
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
            
        return

    # TODO: predic function
    def predict(self, features):
        features = [['a', 'b'], ['b', 'a'], ['b', 'b']]
        predicted_features = []
        for i in features:
            predicted_features.append(self.root_node.predict(i))
        return predicted_features

class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    # TODO: implement split function
    def split(self):
        
        features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['c', 'b']]
        labels = [0, 0, 1, 1]
        
        for current_dim in range(len(features[0])):
            if not 'max_gain' in locals():
                max_gain = -9999
            current_dim_1 = 0    
            current_x = np.array(features)[:, current_dim_1]
            if None in current_x:
                continue
            branch_values = np.unique(current_x)
            
            
            branches = np.zeros((len(branch_values), num_cls + 1))
            
            
            
            #branches = np.zeros((num_cls, len(branch_values)))
            
            
            total_entropy = 0
            C = np.unique(labels)
            for c in C:
                p = float(np.count_nonzero(labels == c)) / len(labels)
                total_entropy += p * np.log2(1/p)
            
           
            
            for i, val in enumerate(branch_values):
                y = np.array(labels)[np.where(current_x == val)]
                for current_y in y:
                    branches[i, current_y] += 1
                    
            
            max_gain_current = Util.Information_Gain(total_entropy, branches)
            if max_gain_current > max_gain:
                max_gain = max_gain_current
                dim_split = current_dim_1
                feature_uniq_split = branch_values.tolist()
        
        
        current_x = np.array(features)[:, dim_split]
        x = np.array(features, dtype = object)
        print(x)
        x[:, dim_split] = None
        
        for i in feature_uniq_split:
            index = np.where(current_x == 'b')
            x_child = x[index].tolist()
            y_child = np.array(labels)[index].tolist()
            child = TreeNode(x_child, y_child, self.num_cls)
            if np.array(x_child).size == 0 or all(x is None for x in x_child[0]):
                child.splittable = False
            self.children.append(child)
            
        for child in self.children:
            if child.splittable:
                child.split()
    
        return
        
        
            
                

    # TODO:treeNode predict function
    def predict(self, feature):
        feature = [['a', 'b'], ['b', 'a'], ['b', 'b']]
        if self.splittable:
            dim_child = feature_uniq_split.index(feature[dim_split])
            return self.children[dim_child].predict(feature)
        else:
            return self.cls_max