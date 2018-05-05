import numpy as np
from scipy import stats
from pprint import pprint
from MNIST import MNIST

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Dense

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

mnist = MNIST()
xtrain, ytrain, xtest, ytest = mnist.load(threeDim=False)

# model = Sequential()
# model.add(Dense(500, input_shape=[784,], activation='relu'))
# model.add(Dense(250, activation='relu'))
# model.add(Dense(125, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

# model.fit(xtrain, ytrain, epochs=10, batch_size=64) # Untrained model

# model.load_weights("model.h5") # Loading pre-trained model
# stats = model.evaluate(xtest, ytest, batch_size=64)
# y_hat = model.predict(xtest)
# accuracy = stats[1] * 100
#
# print (accuracy)

def get_chart_for_model(kNN, trialsX, trialsY):
    accuracy_chart = []
    test_index_X = 0
    test_index_y = 0
    
    for i in range(len(trialsX)):
        print ('Trial {}'.format(i+1))
        
        current_trial_batch_X = trialsX[i] # Each of the 5 trials per model with 5 folds each
        current_trial_batch_y = trialsY[i] # Each of the 5 trials per model with 5 folds each
        print ('Total batch: {} | {}'.format(current_trial_batch_X.shape, current_trial_batch_y.shape))
        
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
        
        # Fitting kNN Model with unified training set and obtaining accuracy by comparing predictions with ground truth labels
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

for kval in range(1, 21):
    kNN = KNeighborsClassifier(n_neighbors=kval)
    print ('For kNN {}'.format(kval))
    trials_X, trials_y = mnist.serve_trials(xtrain, ytrain, xtest, ytest, 5)
    accuracy_chart_for_model = get_chart_for_model(kNN, trials_X, trials_y) # Get accuracy chart of 5 accuracies per trial
    print ('Accuracy chart for kNN {}: {}'.format(kval, accuracy_chart_for_model))
    print ('_________________________________________________________')