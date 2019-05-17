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
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
            
        return

        

    # TODO: predic function
    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred

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
        
        
        
        for current_dim in range(len(self.features[0])):
            if not 'max_gain' in locals():
                max_gain = -9999
                
            current_x = np.array(self.features)[:, current_dim]
            if None in current_x:
                continue
            branch_values = np.unique(current_x)
            if not 'branch_values_current' in locals():
                branch_values_current = -1
            if not 'current_current_dim' in locals():
                current_current_dim = -1
            
            #branches = np.zeros((self.num_cls, len(branch_values)))
            branches = np.zeros((len(branch_values), self.num_cls + 1))
            
            for i, val in enumerate(branch_values):
                y = np.array(self.labels)[np.where(current_x == val)]
                for current_y in y:
                    branches[i, current_y] += 1
                    
            total_entropy = 0
            
            C = np.unique(self.labels)
            for c in C:
                p = float(np.count_nonzero(self.labels == c)) / len(self.labels)
                total_entropy += p * np.log2(1/p)
                
            max_gain_current = Util.Information_Gain(total_entropy, branches)
            
            if max_gain_current == max_gain and branch_values.shape[0] > branch_values_current:
                max_gain = max_gain_current
                self.dim_split = current_dim
                self.feature_uniq_split = branch_values.tolist()
                branch_values_current = branch_values.shape[0]
                current_current_dim = current_dim
            
            if max_gain_current > max_gain:
                max_gain = max_gain_current
                self.dim_split = current_dim
                self.feature_uniq_split = branch_values.tolist()
                branch_values_current = branch_values.shape[0]
                current_current_dim = current_dim
                
        
        current_x = np.array(self.features)[:, self.dim_split]
        x = np.array(self.features, dtype = object)
        x[:, self.dim_split] = None
        
        for i in self.feature_uniq_split:
            index = np.where(current_x == i)
            x_child = x[index].tolist()
            y_child = np.array(self.labels)[index].tolist()
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
        import random
        if self.splittable:
            if feature[self.dim_split] not in self.feature_uniq_split:#.index(feature[self.dim_split]):
                dim_child=random.randint(0,len(self.feature_uniq_split)-1)
            else:
                dim_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[dim_child].predict(feature)
        else:
            return self.cls_max