import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import math
import random
X_0 = pd.read_csv('/Users/Harnoor/Downloads/hw1/data_csv/X.txt',header = None)
Y_0 = pd.read_csv('/Users/Harnoor/Downloads/hw1/data_csv/Y.txt',header = None)
columns = ['intrcpt', 'num cyl', 'displ', 'hp', 'weight', 'accel', 'year']
X_orig.columns = columns
X, y = X_0, y_0

def data_shuffle(X,y):
    i = random.randint(0,len(X)) - 1
    assert X.index[i] == y.index[i], 'Data not aligned'
    rows = list(X.index)
    random.shuffle(rows)
    X = X.reindex(rows)
    y = y.reindex(rows)
    X.index = range(len(X))
    y.index = range(len(y))
    return X,y

def split_data(X, y):
    A = X.iloc[:20] # This is the training data
    a = y.iloc[:20]
    B = X.iloc[20:] # This is the testing data
    b = y.iloc[20:]
    return A,a,b,B

def get_y_hat(w_hat, X):
    return X.dot(w_hat)

def get_MAE(y, y_hat):
    return (y - y_hat).abs().sum() / len(y)

def get_RMSE(y, y_hat):
    return math.sqrt(((y - y_hat) ** 2).sum() / len(y))

def loglikelihood (sample):
    n = len(sample)
    data = sample.values
    mu = sum(data) / float(len(data)) # Unbiased estimator ## Calculating mean
    var = sum([(x - mu) ** 2 for x in data]) / float(len(data)-1) ## Calculating    # variance
    term1 = -(n/2.) * math.log(2 * math.pi)
    term2 = -(n/2.) * math.log(var)
    term3 = -(1 / (2. * var)) * sum([(x - mu) ** 2 for x in sample])
    return term1 + term2 + term3

