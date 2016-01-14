__author__ = 'stgy'
import numpy as np
from utils import sigmoid,corrupt,loadData
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import randint
import time
from layers import *

class SparseAutoEncoder:
    def __init__(self,input_size,hidden_size,W1=None,W2=None,b1=None,b2=None):
        if W1 is None:
            low = -4 * np.sqrt(6.0 / (input_size + hidden_size))
            high = 4 * np.sqrt(6.0 / (input_size + hidden_size))
            self.W1 = np.random.uniform(low,high,size = (input_size,hidden_size))
        else:
            self.W1 = W1

        if W2 is None:
            low = -4 * np.sqrt(6.0 / (input_size + hidden_size))
            high = 4 * np.sqrt(6.0 / (input_size + hidden_size))
            self.W2 = np.random.uniform(low,high,size = (hidden_size,input_size))
        else:
            self.W2 = W2

        if b1 is None:
            self.b1 = np.zeros((1,hidden_size))
        else:
            self.b1 = b1

        if b2 is None:
            self.b2 = np.zeros((1,input_size))
        else:
            self.b2 = b2

    def get_hidden_output(self,x):
        return sigmoid(x.dot(self.W1) + self.b1)

    def train(self,x,epochs = 15,lr = 0.1, batch_size = 20, regularization = 1e-3,beta = 1,activation_level = 0.005):
        n_batch = x.shape[0] / batch_size
        learning_curve_list = []
        for i in xrange(epochs):
            Loss = []
            if i == 0:
                start_time = time.clock()
            for j in xrange(n_batch):
                batch_x = x[j * batch_size: (j + 1) * batch_size]

                hidden_in = batch_x.dot(self.W1) + self.b1
                hidden_out = sigmoid(hidden_in)

                average_activation = np.sum(hidden_out, axis = 0) / batch_size

                reconstruct_in = hidden_out.dot(self.W2) + self.b2
                reconstruct_out = sigmoid(reconstruct_in)

                square_loss = 0.5 * np.sum((reconstruct_out - batch_x) ** 2) / batch_size
                reg_loss = 0.5 * regularization * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
                KL_loss = beta * np.sum((activation_level * np.log(activation_level / average_activation) +
                                         (1 - activation_level) * np.log((1 - activation_level) / (1 - average_activation))))
                total_loss = square_loss + reg_loss + KL_loss
                Loss.append(total_loss)

                grad_reconstruct_out = (reconstruct_out - batch_x) / batch_size
                grad_reconstruct_in = grad_reconstruct_out * reconstruct_out * (1 - reconstruct_out)
                grad_W2 = (hidden_out.T).dot(grad_reconstruct_in)
                grad_b2 = np.sum(grad_reconstruct_in, axis = 0)

                grad_hidden_out = grad_reconstruct_in.dot(self.W2.T) + beta * (
                    (1 - activation_level) / (1 - average_activation) - activation_level / average_activation) / batch_size
                grad_hidden_in = grad_hidden_out * (hidden_out) * (1 - hidden_out)

                grad_W1 = (batch_x.T).dot(grad_hidden_in)
                grad_b1 = np.sum(grad_hidden_in, axis=0)

                self.W2 -= lr * (grad_W2 + regularization * self.W2)
                self.W1 -= lr * (grad_W1 + regularization * self.W1)
                self.b1 -= lr * (grad_b1)
                self.b2 -= lr * (grad_b2)
            mean_loss = np.mean(Loss)
            learning_curve_list.append(mean_loss)
            print "average loss is: %f" % mean_loss
            if i % 10 == 0:
                cmap = mpl.cm.gray_r
                norm = mpl.colors.Normalize(vmin=0)

                rand_index = randint(0,x.shape[0])
                plt.subplot(1,2,1)
                plt.imshow(x[rand_index].reshape(28,28),cmap = cmap)
                plt.subplot(1,2,2)
                hidden_random = sigmoid(x[rand_index].dot(self.W1) + self.b1)
                recons_random = sigmoid(hidden_random.dot(self.W2) + self.b2)
                plt.imshow(recons_random.reshape(28,28),cmap = cmap)
                plt.show()

                for j in xrange(100):
                    plt.subplot(10,10,j)
                    plt.axis('off')
                    plt.imshow(self.W1.T[j,:].reshape(28,28),cmap = cmap)
                plt.show()
            if i == 0:
                stop_time = time.clock()
                print "one single epoch runs %f minutes!" % ((stop_time - start_time) / 60.0)

        plt.plot(learning_curve_list)
        plt.show()

class DenoisingAutoEncoder:
    '''initlize weights and bias'''
    def __init__(self,input_size,hidden_size,W1=None,b1=None,b2=None):
        if W1 is None:
            low = -4 * np.sqrt(6.0 / (input_size + hidden_size))
            high = 4 * np.sqrt(6.0 / (input_size + hidden_size))
            self.W1 = np.random.uniform(low,high,size = (input_size,hidden_size))
        else:
            self.W1 = W1

        if b1 is None:
            self.b1 = np.zeros((1,hidden_size))
        else:
            self.b1 = b1

        if b2 is None:
            self.b2 = np.zeros((1,input_size))
        else:
            self.b2 = b2

        self.W2 = self.W1.T # tie weight


    def get_hidden_output(self,x):
        return sigmoid(x.dot(self.W1) + self.b1)

    def train(self,x,epochs = 15,lr = 0.01, batch_size = 20, corruption_level = 0.3, regularization = 0):
        n_batch = x.shape[0] / batch_size
        corrupt_x = corrupt(x,corruption_level)    # add noise to the original data
        learning_curve_list = []

        for i in xrange(epochs):
            Loss = []
            if i == 0:
                start_time = time.clock()
            for j in xrange(n_batch):
                batch_x = x[j * batch_size : (j + 1) * batch_size]   # get minibatch of original data
                corrupt_batch_x = corrupt_x[j * batch_size : (j + 1) * batch_size]  # get minibatch of corrupted data

                hidden_in,cache1 = affine_forward(corrupt_batch_x,self.W1,self.b1)
                hidden_out,cache2 = sigmoid_forward(hidden_in)
                reconstruct_in,cache3 = affine_forward(hidden_out,self.W2,self.b2)

                batch_loss,dscore = cross_entropy_loss(reconstruct_in,batch_x)
                Loss.append(batch_loss)

                """forward"""
                grad_W2,grad_b2,grad_hidden_out = affine_backward(dscore,cache3)
                grad_hidden_in = sigmoid_backward(grad_hidden_out,cache2)
                grad_W1,grad_b1,_ = affine_backward(grad_hidden_in,cache1)

                """back_propagation"""
                self.W1 -= lr * (grad_W1 + grad_W2.T + regularization * self.W1)
                self.b1 -= lr * (grad_b1)
                self.b2 -= lr * (grad_b2)

            mean_loss = np.mean(Loss)
            learning_curve_list.append(mean_loss)
            print "average loss is: %f" % mean_loss

            '''visualize weight'''
            if i % 10 == 0:
                cmap = mpl.cm.gray_r
                norm = mpl.colors.Normalize(vmin=0)

                rand_index = randint(0,x.shape[0])
                plt.subplot(1,3,1)
                plt.imshow(x[rand_index].reshape(28,28),cmap = cmap)
                plt.subplot(1,3,2)
                plt.imshow(corrupt_x[rand_index].reshape(28,28),cmap = cmap)
                hidden_random = sigmoid(corrupt_x[rand_index].dot(self.W1) + self.b1)
                recons_random = sigmoid(hidden_random.dot(self.W2) + self.b2)
                plt.subplot(1,3,3)
                plt.imshow(recons_random.reshape(28,28),cmap = cmap)
                plt.show()

                for i in xrange(100):
                    plt.subplot(10,10,i)
                    plt.axis('off')
                    plt.imshow(self.W1.T[i,:].reshape(28,28),cmap = cmap)
                plt.show()
            if i == 0:
                stop_time = time.clock()
                print "one single epoch runs %i minutes!" % ((stop_time - start_time) / 60.0)

        plt.plot(learning_curve_list)
        plt.show()


if __name__ == "__main__":
    dataset = "mnist.pkl.gz"
    train_set, valid_set, test_set = loadData(dataset)
    train_x,train_y = train_set
    valid_x,valid_y = valid_set
    test_x,test_y = test_set
    print "the size of training set is:(%d,%d)" % train_x.shape

    n_sample,feature_size = train_x.shape
    n_hidden = 500
    epochs = 100

    lr = 0.1
    batch_size = 20
    corruption_level = 0.3
    regularization = 0
    print "initializing AutoEncoder......"
    dA = DenoisingAutoEncoder(feature_size,n_hidden)
    print "start training......"
    dA.train(train_x,epochs,lr,batch_size,corruption_level,regularization)
    print "finish training!"

    '''lr = 0.1
    batch_size = 100
    regularization = 1e-4
    activation_level = 0.01
    beta = 3
    print "initializing AutoEncoder......"
    dA = SparseAutoEncoder(feature_size,n_hidden)
    print "start training......"
    dA.train(train_x,epochs,lr,batch_size,regularization,beta,activation_level)
    print "finish training!"'''
