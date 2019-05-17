"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    #import math
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    y_pred = np.matmul(X,w)
    diff = (y_pred - y)**2
    err = np.mean(diff)
    #err = (math.ceil(err*1e9)/1e9)
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    w_1 = np.linalg.inv(np.matmul(np.transpose(X),X))
    w_2 = np.matmul(w_1,np.transpose(X))
    w_3 = np.matmul(w_2,y)
    w = w_3
    return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    import math
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    w_1 = np.matmul(np.transpose(X),X)
    eigen_values, v_1 = np.linalg.eig(w_1)
    smallest_eigen_value = np.amin(eigen_values)
    #print(smallest_eigen_value)
    while smallest_eigen_value < math.pow(10, -5):
        w_1 = np.matmul(np.transpose(X),X)
        identity_matrix = math.pow(10, -1) * np.identity(np.size(X, 1))
        w_1 = w_1 + identity_matrix
        eigen_values, v_1 = np.linalg.eig(w_1)
        smallest_eigen_value = np.amin(eigen_values)
        #print(smallest_eigen_value)
    w_1 = np.linalg.inv(w_1)
    w_2 = np.matmul(w_1, np.transpose(X))
    w_3 = np.matmul(w_2, y)
    w = w_3
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    identity_matrix = lambd * np.identity(np.size(X, 1))
    w_1 = np.linalg.inv(np.matmul(np.transpose(X),X) + identity_matrix)
    w_2 = np.matmul(w_1, np.transpose(X))
    w_3 = np.matmul(w_2, y)
    w = w_3
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    import math
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    x = -19
    lambd = math.pow(10, x)
    bestlambda = -1e30
    bestmse = 1e30
    while x != 20:
        #print (lambd)
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        mse = mean_square_error(w, Xval, yval)
        #print(mse)
        if mse < bestmse:
            bestmse = mse
            bestlambda = lambd
        x = x + 1
        lambd = math.pow(10, x)
    
    
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    
    X_temp = X
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    for x in range(2, power + 1):
        X_new = np.power(X_temp, x)
        #X_new = np.square(X)
        X_new_ = np.append(X, X_new, axis = 1)
        X = X_new_
    
    return X


