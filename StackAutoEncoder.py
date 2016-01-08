__author__ = 'stgy'

import numpy as np
from utils import sigmoid,softmax,neg_log_likelihood,loadData
from AutoEncoder import DenoisingAutoEncoder

class mlp:
    def __init__(self,input_size,hidden_layers_size,output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_size = hidden_layers_size
        self.n_hidden = len(hidden_layers_size)

        self.W = []
        self.b = []
        for i in xrange(self.n_hidden + 1):
            if i == 0:
                fan_in = input_size
                fan_out = hidden_layers_size[i]
            elif (i == len(hidden_layers_size)):
                fan_in = hidden_layers_size[-1]
                fan_out = output_size
            else:
                fan_in = hidden_layers_size[i - 1]
                fan_out = hidden_layers_size[i]
            low = -4 * np.sqrt(6.0 / (fan_in + fan_out))
            high = 4 * np.sqrt(6.0 / (fan_in + fan_out))
            w = np.random.uniform(low,high,size = (fan_in,fan_out))
            b = np.zeros((1,fan_out))
            self.W.append(w)
            self.b.append(b)


    def pre_training(self,train_x,pre_train_epochs = 15,pre_lr = 0.001,batch_size = 20,corruption_levels = [0.1,0.2,0.3],regularization = 0.0):
        dA_container = []
        dA_input = None
        for i in xrange(self.n_hidden):
            if i == 0:
                dA_input = train_x
                dA = DenoisingAutoEncoder(self.input_size,self.hidden_layers_size[i],self.W[i],self.b[i],b2 = None)
            else:
                dA_input = dA_container[-1].get_hidden_output(dA_input)
                dA = DenoisingAutoEncoder(self.hidden_layers_size[i - 1],self.hidden_layers_size[i],self.W[i],self.b[i],b2 = None)
            print "dA " + str(i + 1) + " start pre_training......"
            corruption_level = corruption_levels[i]
            dA.train(dA_input,pre_train_epochs,pre_lr,batch_size,corruption_level,regularization)
            dA_container.append(dA)
            print "dA " + str(i + 1) + " has finished pre_training!"


    def minibatch_update(self,x,y,lr,regularization):
        n_sample = x.shape[0]
        info = x
        hidden_cache = []
        for i in xrange(self.n_hidden + 1):
            if i == self.n_hidden:
                probs = softmax(info.dot(self.W[i]) + self.b[i])
            else:
                info = sigmoid(info.dot(self.W[i]) + self.b[i])
                hidden_cache.append(info)
        loss = neg_log_likelihood(probs,y)
        probs[np.arange(n_sample),y] -= 1.0
        errors = probs
        for i in range(self.n_hidden,-1,-1):
            if i >= 1:
                hidden_out = hidden_cache[i - 1]
                grad_hidden_out = errors.dot(self.W[i].T)
                self.W[i] -= (lr * (hidden_out.T).dot(errors) + regularization * self.W[i])
                self.b[i] -= lr * np.sum(errors,axis = 0)
                errors = hidden_out * (1 - hidden_out) * grad_hidden_out
            else:
                hidden_out = x
                self.W[i] -= (lr * (hidden_out.T).dot(errors) + regularization * self.W[i])
                self.b[i] -= lr * np.sum(errors,axis = 0)

        return loss


    def compute_loss(self,x,y):
        info = x
        for i in xrange(self.n_hidden + 1):
            if i == self.n_hidden:
                scores = info.dot(self.W[i]) + self.b[i]
            else:
                info = sigmoid(info.dot(self.W[i]) + self.b[i])
        y_given_x = np.argmax(scores,axis = 1)
        error = np.mean(y_given_x != y)
        return error


    def fine_tuning(self,train_data,validation_data,test_data,fine_tune_epochs = 100,fine_tune_lr = 0.1,regularization = 0.0,batch_size = 20):
        train_x,train_y = train_data
        validation_x,validation_y = validation_data
        test_x,test_y = test_data
        n_train_batch = train_x.shape[0] / batch_size
        patience = 10 * n_train_batch
        patience_increase = 2.0
        improvement_threshhold = 0.995
        validation_frequency = min(n_train_batch,patience / 2)
        best_validation_error = np.inf
        test_score = 0.
        done_looping = False
        epoch = 0

        while (epoch < fine_tune_epochs) and (not done_looping):
            epoch += 1
            for i in xrange(n_train_batch):
                iter = (epoch - 1) * n_train_batch + i
                x = train_x[i * batch_size:(i + 1) * batch_size]
                y = train_y[i * batch_size:(i + 1) * batch_size]
                minibatch_avg_cost = self.minibatch_update(x,y,fine_tune_lr,regularization)

                if (iter + 1) % validation_frequency == 0:
                    validation_errors = self.compute_loss(validation_x,validation_y)
                    this_validation_error = np.mean(validation_errors)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, i + 1, n_train_batch,
                       this_validation_error * 100.))

                    if (this_validation_error < best_validation_error):
                        if (this_validation_error < best_validation_error * improvement_threshhold):
                            patience = max(patience, iter * patience_increase)
                        best_validation_error = this_validation_error
                        best_iter = iter

                        test_errors = self.compute_loss(test_x,test_y)
                        test_score = np.mean(test_errors)

                        print(('    epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, i + 1, n_train_batch,
                           test_score * 100.))
                if patience <= iter:
                    done_looping = True
                    break

        print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_error * 100., best_iter + 1, test_score * 100.)
    )

if __name__ == "__main__":
    dataset = "mnist.pkl.gz"
    train_set, valid_set, test_set = loadData(dataset)
    train_x,train_y = train_set
    valid_x,valid_y = valid_set
    test_x,test_y = test_set

    input_feature_size = train_x.shape[1]
    output_size = 10
    neuralNet = mlp(input_feature_size,[500,500,500],10)

    pre_train_epochs = 15
    pre_train_lr = 0.001
    pre_train_batch_size = 20
    pre_train_corruption_levels = [0.1,0.2,0.3]
    pre_train_regularization = 0.0
    neuralNet.pre_training(train_x,pre_train_epochs,pre_train_lr,pre_train_batch_size,pre_train_corruption_levels,pre_train_regularization)

    fine_tune_epochs = 100
    fine_tune_lr = 0.1
    fine_tune_regularization = 0.0
    fine_tune_batch_size = 5
    neuralNet.fine_tuning(train_set,valid_set,test_set,fine_tune_epochs,fine_tune_lr,fine_tune_regularization,fine_tune_batch_size)



