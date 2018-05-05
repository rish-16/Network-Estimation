import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

class MNIST():
    def load(self, threeDim=False):
        (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
        
        if not threeDim:
            xtrain = xtrain.reshape([len(xtrain), 784]) / 255
            xtest = xtest.reshape([len(xtest), 784]) / 255
            
            ytrain = to_categorical(ytrain, 10)
            ytest = to_categorical(ytest, 10)
        else:
            xtrain = xtrain.reshape([len(xtrain), 28, 28, 1]) / 255
            xtest = xtest.reshape([len(xtest), 28, 28, 1]) / 255
        
            ytrain = to_categorical(ytrain, 10)
            ytest = to_categorical(ytest, 10)
            
        return xtrain, ytrain, xtest, ytest
        
    def permute_set(self, xtrain, ytrain, xtest, ytest, folds):
        X = []
        y = []
        X.append(xtrain)
        X.append(xtest)
        y.append(ytrain)
        y.append(ytest)
        
        X = np.concatenate((xtrain, xtest)).reshape([len(xtrain)+len(xtest), 784])
        y = np.concatenate((ytrain, ytest))
        
        complete_trial_X = np.array_split(X, folds) # Complete trial consisting of 5 folds on the dataset
        complete_trial_y = np.array_split(y, folds) # Complete trial consisting of 5 folds on the dataset
        
        trials_X = []
        trials_y = []
        
        for i in range(folds):
            trials_X.append(complete_trial_X)
            trials_y.append(complete_trial_y)
        
        trials_X = np.array(trials_X) # Complete set of 5 trials per model for all 20 kNN models
        trials_y = np.array(trials_y) # Complete set of 5 trials per model for all 20 kNN models
        
        print ('{} and {}'.format(trials_X.shape, trials_y.shape)) # All 20 * 5 trials comprising of 5 folds per trial
        
        return trials_X, trials_y
    