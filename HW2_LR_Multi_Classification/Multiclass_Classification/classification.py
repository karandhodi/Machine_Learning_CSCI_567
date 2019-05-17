from __future__ import division, print_function
import numpy as np
import bm_classify as sol
from sklearn.datasets import make_classification, make_blobs, make_moons, load_iris
from sklearn.model_selection import train_test_split

import json
import pandas as pd


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)

def toy_data_binary():
  data = make_classification(n_samples=500, 
                              n_features=2,
                              n_informative=1, 
                              n_redundant=0, 
                              n_repeated=0, 
                              n_classes=2, 
                              n_clusters_per_class=1, 
                              class_sep=1., 
                              random_state=42)

  X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.7, random_state=42)
  return X_train, X_test, y_train, y_test

def moon_dataset():
  data = make_moons(n_samples=500, shuffle=True, noise=0.2, random_state=42)
  X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.7, random_state=42)
  return X_train, X_test, y_train, y_test


# Multiple classification data

def smile_dataset_clear():
  data = make_blobs(n_samples=870,
            n_features=2,
            random_state=42, 
            centers=[[-5, 5], [5, 5], [-4, -2], [-2, -4], [0, -5], [2, -4], [4, -2] ],
            cluster_std=1)
  data[1][data[1] > 2] = 2
  X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.7, random_state=42)
  return X_train, X_test, y_train, y_test

def smile_dataset_blur():
  data = make_blobs(n_samples=50000,
            n_features=2,
            random_state=42, 
            centers=[[-5, 5], [5, 5], [-4, -2], [-2, -4], [0, -5], [2, -4], [4, -2] ],
            cluster_std=2)
  data[1][data[1] > 2] = 2
  X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], train_size=0.7, random_state=42)
  return X_train, X_test, y_train, y_test


# Hand-written digits data
def data_loader_mnist(dataset='mnist_subset.json'):
  # This function reads the MNIST data and separate it into train, val, and test set
  with open(dataset, 'r') as f:
        data_set = json.load(f)
  train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

  return np.asarray(train_set[0]), \
          np.asarray(test_set[0]), \
          np.asarray(train_set[1]), \
          np.asarray(test_set[1])

def run_binary():
    from data_loader import toy_data_binary, \
                            moon_dataset, \
                            data_loader_mnist 

    datasets = [(toy_data_binary(), 'Synthetic data'), 
                (moon_dataset(), 'Two Moon data'),
                (data_loader_mnist(), 'Binarized MNIST data')]

    for data, name in datasets:
        print(name)
        X_train, X_test, y_train, y_test = toy_data_binary()

        if name == 'Binarized MNIST data':
            y_train = [0 if yi < 5 else 1 for yi in y_train]
            y_test = [0 if yi < 5 else 1 for yi in y_test]
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

        for loss_type in ["perceptron", "logistic"]:
            w, b = sol.binary_train(X_train, y_train, loss="logistic")
            train_preds = sol.binary_predict(X_train, w, b, loss=loss_type)
            preds = sol.binary_predict(X_test, w, b, loss=loss_type)
            print(loss_type + ' train acc: %f, test acc: %f' 
                %(accuracy_score(y_train, train_preds), accuracy_score(y_test, preds)))
        print()

def run_multiclass():
    from data_loader import smile_dataset_clear, \
                            smile_dataset_blur, \
                            data_loader_mnist 
    import time
    datasets = [(smile_dataset_clear(), 'Clear smile data', 3) 
                ,(smile_dataset_blur(), 'Blur smile data', 3)
                ,(data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = smile_dataset_clear()
        for gd_type in ["sgd", "gd"]:
            s = time.time()
            w, b = sol.multiclass_train(X_train, y_train, C=3, gd_type='gd')
            print(gd_type + ' training time: %0.6f seconds' %(time.time()-s))
            train_preds = sol.multiclass_predict(X_train, w=w, b=b)
            preds = sol.multiclass_predict(X_test, w=w, b=b)
            print('train acc: %f, test acc: %f' 
                % (accuracy_score(y_train, train_preds), accuracy_score(y_test, preds)))
        print()
        


if __name__ == '__main__':
    
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
