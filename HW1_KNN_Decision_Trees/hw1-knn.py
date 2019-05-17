from __future__ import division, print_function

from typing import List, Callable

import numpy as np
import scipy
from collections import Counter


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

'''x = np.array([[3, 1, 2],[1,1,1]])
y = x.shape[0]
ass = np.argsort(x)[:4]
asp = np.argpartition(x, 0)'''
x = [[1,2,3],[3,4,5]]
y = len(x)

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: Complete the training function
    def train(self, features: List[List[float]], labels: List[int]):
        #raise NotImplementedError
        self.features = features
        self.labels = labels



    
    #TODO: Complete the prediction function
    def predict(self, features: List[List[float]]) -> List[int]:
        #raise NotImplementedError
        predicted_labels = []
        
        for i in features:
            
            distances = []
            
            for j in self.features:
                distances.append(self.distance_function(i, j))
            
            #indices = np.argpartition(distances, self.k)
            indices = np.argsort(distances)[:self.k]
            
            votes_0 = 0
            votes_1 = 0
            
            for x in range(0, self.k):
                current_label = self.labels[indices[x]]
                if current_label == 0:
                    votes_0 += 1
                else:
                    votes_1 += 1
            
            if votes_0 > votes_1:
                predicted_labels.append(0)
            else:
                predicted_labels.append(1)
        
        return predicted_labels
            
           
                
            
    #TODO: Complete the get k nearest neighbor function
    def get_k_neighbors(self, point):
        X = 1
        #raise NotImplementedError
        
    #TODO: Complete the model selection function where you need to find the best k     
    def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, f1_score, Xval, yval, Xtest, ytest):
        
            best_f1_score = -1
            best_k = 0
            best_distance_func_name = {}
            
            for name, dist_func in distance_funcs.items():
                
                
                for k in range(1, 31, 2):
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
            model = KNN(k = best_k, distance_function = distance_funcs.get(best_distance_func_name))
            model.train(np.concatenate((Xtrain, Xval),axis = 0), np.concatenate((ytrain, yval),axis = 0))
            test_f1_score = f1_score(ytest, model.predict(Xtest))
            name = best_distance_func_name
            '''
            print()
            print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k) +
                  'test f1 score: {test_f1_score:.5f}'.format(test_f1_score=test_f1_score))
            print()
            '''
                
            return best_k, model, best_distance_func_name
    
    #TODO: Complete the model selection function where you need to find the best k with transformation
    def model_selection_with_transformation(distance_funcs,scaling_classes, Xtrain, ytrain, f1_score, Xval, yval, Xtest, ytest):
        
        best_f1_score = -1
        best_k = 0
        best_distance_func_name = {}   
        
        if distance_funcs is not None and scaling_classes is not None:
            for scaling_name, scaling_function in scaling_classes.items():
                for name, dist_func in distance_funcs.items():
                    scaler = scaling_function()
                    train_scaled = scaler(Xtrain)
                    valid_scaled = scaler(Xval)
                    for k in range(1, 31, 2):
                        model = KNN(k = k, distance_function = dist_func)
                        model.train(train_scaled, ytrain)
                        train_f1_score = f1_score(ytrain, model.predict(train_scaled))
                        
                        valid_f1_score = f1_score(yval, model.predict(valid_scaled))
                    
                        if valid_f1_score > best_f1_score:
                            best_f1_score = valid_f1_score
                            best_k = k
                            best_distance_func_name = name
                            
                    
                        '''
                        #Dont change any print statement
                        print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
                                  'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) + 
                                  'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
                        '''
                    
                scaler = scaling_function()
                combined_scaled = scaler(np.concatenate((Xtrain, Xval),axis = 0))
                test_scaled = scaler(Xtest)
                
                model = KNN(k = best_k, distance_function = distance_funcs.get(best_distance_func_name))
                model.train(combined_scaled, np.concatenate((ytrain, yval),axis = 0))
                test_f1_score = f1_score(ytest, model.predict(test_scaled))
                name = best_distance_func_name
            
                '''
                print()
                print('[part 1.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
                      'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k, test_f1_score=test_f1_score))
                print()
                '''
                
        return best_k, model
        
        
    #TODO: Do the classification 
    def test_classify(model):
        from data import test_processing
        Xtest = test_processing()
        model.predict(Xtest)
        

if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
