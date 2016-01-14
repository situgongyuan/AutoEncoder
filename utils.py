__author__ = 'stgy'
import numpy as np
from scipy import stats
import gzip
import cPickle

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def neg_log_likelihood(probs,target):
    return -np.mean(np.log(probs[np.arange(target.shape[0]),target]))

def softmax(X):
    num_of_samples = X.shape[0]
    scores = X - np.max(X,axis = 1,keepdims=True)
    scores = np.exp(scores)
    probability = scores / (np.sum(scores,axis = 1,keepdims=True))
    #lossVector = -np.log(probability[range(num_of_samples),y])
    #loss = np.sum(lossVector) / num_of_samples
    return probability

def corrupt(X,level):
    mask = stats.bernoulli.rvs(1.0 - level, size = X.shape)
    return X * mask

def loadData(dataset):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set