import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from scipy import stats

class PMNIST():
    def load_pmnist(self, threeDim=False):
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
        
    def serve_trials(self, xtrain, ytrain, xtest, ytest, folds):
        X = []
        y = []
        X.append(xtrain)
        X.append(xtest)
        y.append(ytrain)
        y.append(ytest)
        
        X = np.concatenate((xtrain, xtest)).reshape([len(xtrain)+len(xtest), 784])
        y = np.concatenate((ytrain, ytest)).reshape([len(xtrain)+len(xtest), 10])
        
        complete_trial_X = np.array_split(X, folds) # Complete trial consisting of 5 folds on the dataset
        complete_trial_y = np.array_split(y, folds) # Complete trial consisting of 5 folds on the dataset
        
        trials_X = []
        trials_y = []
        
        for i in range(folds):
            trials_X.append(complete_trial_X) # Appending 5 folds per trial for 5 trials for 20 kNN models
            trials_y.append(complete_trial_y) # Appending 5 folds per trial for 5 trials for 20 kNN models
        
        trials_X = np.array(trials_X) # Complete set of 5 trials per model for all 20 kNN models
        trials_y = np.array(trials_y) # Complete set of 5 trials per model for all 20 kNN models
        
        return trials_X, trials_y
        
    def get_chart_for_model(self, kNN, trialsX, trialsY):
        accuracy_chart = []
        test_index_X = 0
        test_index_y = 0
        for trial in range(len(trialsX)):
            print ('Trial {}'.format(trial+1))
            
            current_trial_batch_X = trialsX[trial] # Each of the 5 trials per model with 5 folds each
            current_trial_batch_y = trialsY[trial] # Each of the 5 trials per model with 5 folds each
            print ('Total Trial batch: {} | {}'.format(current_trial_batch_X.shape, current_trial_batch_y.shape))
            
            # Per trial analysis
            x_test_fold = current_trial_batch_X[test_index_X] # Test set X for each trial
            y_test_fold = current_trial_batch_y[test_index_y] # Test set y for each trial
            print ('Testing batch: {} | {}'.format(x_test_fold.shape, y_test_fold.shape))
            
            x_train_folds = np.delete(current_trial_batch_X, test_index_X, 0) # The current trial batch without the test fold
            y_train_folds = np.delete(current_trial_batch_y, test_index_y, 0) # The current trial batch without the test fold
            
            # Merge training folds to unified array
            x_train_fold = np.concatenate((x_train_folds))
            y_train_fold = np.concatenate((y_train_folds))
            print ('Training batch: {} | {}'.format(x_train_fold.shape, y_train_fold.shape))
            
            # Fitting current kNN Model with unified training set and obtaining accuracy by comparing predictions with ground truth
            kNN.fit(x_train_fold, y_train_fold)
            y_hat_fold = kNN.predict(x_test_fold)
            accuracy = accuracy_score(y_hat_fold, y_test_fold)
            print ('Accuracy for Trial {}: {}'.format(test_index_X+1, accuracy))
            accuracy_chart.append(accuracy)
            
            test_index_X += 1 # Increment index by 1 to change test set X for next trial
            test_index_y += 1 # Increment index by 1 to change test set y for next trial
            print ()
            
        accuracy_chart = np.array(accuracy_chart)
        
        return accuracy_chart
        
    def get_pval_for_charts(self, Ac1, Ac2):
        np.random.seed(0)
        pval = stats.ttest_rel(Ac1, Ac2)[1]
        
        return pval
        
    