import numpy as np
from typing import List
from hw1_knn import KNN
import math


np.seterr(divide='ignore', invalid='ignore')

#TODO: Information Gain function
def Information_Gain(S, branches):
    # branches: List[List[any]]
    # return: float
    
    branches = np.array(branches)
    branches_1 = branches.transpose()
    
    totals = np.sum(branches_1, axis=0)
    fractions = totals / np.sum(totals)
    current_entropy = branches_1 / totals
    
    a = 0
    for x in current_entropy:
        b = 0
        for i in x:
            
            if i > 0:
                current_entropy[a][b] = np.array([-i * np.log2(i)])
                b += 1
            else:
                current_entropy[a][b] = 0
                b += 1
        a += 1
            
    current_entropy = np.sum(current_entropy, axis=0)
    current_entropy = np.sum(current_entropy * fractions)
    
    IG = S - current_entropy
    return IG

# TODO: implement reduced error pruning
def reduced_error_pruning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List[any]
    X = 1


# print current tree
# Do not change this function
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    for idx_cls in range(node.num_cls):
        string += str(node.labels.count(idx_cls)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#KNN Utils

#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    
    tp = 0
    fp = 0
    fn = 0
    
    for i in range(len(real_labels)):
        if real_labels[i] == 1 and predicted_labels[i] == 1:
            tp += 1
        if real_labels[i] == 0 and predicted_labels[i] == 1:
            fp += 1
        if real_labels[i] == 1 and predicted_labels[i] == 0:
            fn += 1
            
    if (2 * tp) + fn + fp == 0:
        return 0
    
    f1_score = (2 * tp)/float(((2 * tp) + fn + fp))
    
    return f1_score
        
    
    #raise NotImplementedError
    
    
#TODO: Euclidean distance, inner product distance, gaussian kernel distance and cosine similarity distance

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    distance = 0
    for i in range(len(point1)):
        distance += pow((point1[i] - point2[i]), 2)
    return math.sqrt(distance)
    #raise NotImplementedError


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    distance = 0
    for i in range(len(point1)):
        distance += point1[i] * point2[i]
    return distance
    #raise NotImplementedError


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    distance = 0
    for i in range(len(point1)):
        distance += pow((point1[i] - point2[i]), 2)
    distance1 = -np.exp(-0.5 * distance)
    return distance1
    #raise NotImplementedError


def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    distance = np.dot(point1, point2)/(np.linalg.norm(point1)*np.linalg.norm(point2))
    return 1 - distance
    #raise NotImplementedError
    
    
# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    best_f1_score = -1
    best_k = 0
    best_distance_func_name = {}
    
    for name, dist_func in distance_funcs.items():
        
        
        for k in range(1, 31, 2):
            if len(Xtrain) < k:
                break
            model = KNN(k = k, distance_function = dist_func)
            
            model.train(Xtrain, ytrain)
            train_f1_score = f1_score(ytrain, model.predict(Xtrain))
            
            valid_f1_score = f1_score(yval, model.predict(Xval))
        
            if valid_f1_score > best_f1_score:
                best_f1_score = valid_f1_score
                best_k = k
                best_distance_func_name = name

            '''
            #Dont change any print statement
            print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) + 
                      'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                      'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
            '''
    #print(best_k, best_distance_func_name)
    best_model = KNN(k = best_k, distance_function = distance_funcs.get(best_distance_func_name))
    best_model.train(np.concatenate((Xtrain, Xval),axis = 0), np.concatenate((ytrain, yval),axis = 0))
    
    '''
    model = KNN(k = best_k, distance_function = distance_funcs.get(best_distance_func_name))
    model.train(np.concatenate((Xtrain, Xval),axis = 0), np.concatenate((ytrain, yval),axis = 0))
    test_f1_score = f1_score(ytest, model.predict(Xtest))
    name = best_distance_func_name
    print()
    print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k) +
          'test f1 score: {test_f1_score:.5f}'.format(test_f1_score=test_f1_score))
    print()'''
        
    return best_model, best_k, best_distance_func_name
    #raise NotImplementedError


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    best_f1_score = -1
    best_k = 0
    best_distance_func_name = {}   
    best_scaler_name = {}
    
    if distance_funcs is not None and scaling_classes is not None:
        for scaling_name, scaling_function in scaling_classes.items():
            for name, dist_func in distance_funcs.items():
                scaler = scaling_function()
                train_scaled = scaler(Xtrain)
                valid_scaled = scaler(Xval)
                for k in range(1, 31, 2):
                    if len(Xtrain) < k:
                        break
                    model = KNN(k = k, distance_function = dist_func)
                    model.train(train_scaled, ytrain)
                    train_f1_score = f1_score(ytrain, model.predict(train_scaled))
                    
                    valid_f1_score = f1_score(yval, model.predict(valid_scaled))
                
                    if valid_f1_score > best_f1_score:
                        best_f1_score = valid_f1_score
                        best_k = k
                        best_distance_func_name = name
                        best_scaler_name = scaling_name
                    '''    
                    print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
                              'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) + 
                              'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
                    '''    
    best_model = KNN(k = best_k, distance_function = distance_funcs.get(best_distance_func_name))
    scaler = scaling_classes.get(best_scaler_name)()
    combined_scaled = scaler(np.concatenate((Xtrain, Xval),axis = 0))
    best_model.train(combined_scaled, np.concatenate((ytrain, yval),axis = 0))
    
    #print(best_k, best_distance_func_name)
    #print(best_scaler_name)
        
            #Dont change any print statement
    '''
                
                
    scaler = scaling_function()
    combined_scaled = scaler(np.concatenate((Xtrain, Xval),axis = 0))
    test_scaled = scaler(Xtest)
    
    model = KNN(k = best_k, distance_function = distance_funcs.get(best_distance_func_name))
    model.train(combined_scaled, np.concatenate((ytrain, yval),axis = 0))
    test_f1_score = f1_score(ytest, model.predict(test_scaled))
    name = best_distance_func_name

    
    print()
    print('[part 1.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
          'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k, test_f1_score=test_f1_score))
    print()'''
            
    return best_model, best_k, best_distance_func_name, best_scaler_name
    #raise NotImplementedError

    
class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized_features = []
        
        for i in features:
            if all(j == 0 for j in i):
                normalized_features.append(i)
            else:
                inner_product = np.sqrt(inner_product_distance(i, i))
                norm = [j / inner_product for j in i]
                normalized_features.append(norm)
                
        return normalized_features
        #raise NotImplementedError
        
"""features = [[3, 4], [1, -1], [0, 0]]
normalized_features = []
for i in features:
    if all(j == 0 for j in i):
        normalized_features.append(i)
    else:
        inner_product = np.sqrt(inner_product_distance(i, i))
        norm = [j / inner_product for j in i]
        normalized_features.append(norm)"""
"""a = [1,2,3]
b = [4,5,6]
distance = 1 - spatial.distance.cosine(a, b)

cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

distance1 = 0
for i in range(len(a)):
    distance1 += a[i] * b[i]

x = 0
for i in range(len(a)):
    x += pow(a[i],2)
x = math.sqrt(x)
y = 0
for i in range(len(b)):
    y += pow(b[i],2)
y = math.sqrt(y)
distance1 = distance1/(x*y)"""



class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.
    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).
    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]
        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]
        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]
        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """

    def __init__(self):
        self.min, self.max = None, None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        feature_array = np.array(features)
        
        if self.min is None or self.max is None:
            self.min = np.amin(feature_array, axis = 0)
            self.max = np.amax(feature_array, axis = 0)
            
        normalized_features = (feature_array - self.min) / (self.max - self.min)
        return normalized_features.tolist()
        #raise NotImplementedError

        
