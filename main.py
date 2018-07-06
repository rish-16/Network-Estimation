import numpy as np
from scipy import stats
from pprint import pprint
from MNIST import PMNIST

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from benchmark import load_mnist, load_model_CNN, train_model

# CNN
xtrain, ytrain, xtest, ytest = load_mnist()
model = load_model_CNN()
trained_model, model_dict = train_model(model, xtrain, ytrain, xtest, ytest)
print (model_dict['acc'])

y_hat_train = trained_model.predict(xtrain)
y_hat_test = trained_model.predict(xtest)

# k-NN
accuracy_charts = []
pmnist = PMNIST()
xtrain, ytrain, xtest, ytest = pmnist.load_pmnist(threeDim=False)

for kval in range(1, 21):
    kNN = KNeighborsClassifier(n_neighbors=kval) # Generate kNN model with different k value
    
    print ('For kNN {}'.format(kval))
    trials_X, trials_y = pmnist.serve_trials(xtrain, np.round(y_hat_train), xtest, np.round(y_hat_test), 5)
    accuracy_chart_for_model = pmnist.get_chart_for_model(kNN, trials_X, trials_y) # Get accuracy chart of 5 accuracies per trial
    accuracy_charts.append(accuracy_chart_for_model)
    
    print ('Accuracy chart for kNN {}: {}'.format(kval, accuracy_chart_for_model))
    print ('_________________________________________________________')
    
accuracy_charts = np.array(accuracy_charts)
Pvals = []
    
for i in range(len(accuracy_charts)):
    base_chart = accuracy_charts[i]
    pvals_for_base_chart = []
    for j in range(len(accuracy_charts)):
        ref_chart = accuracy_charts[i][j]
        pval_with_ref_chart = pmnist.get_pval_for_charts(base_chart, ref_chart) # Obtaining p value using paired t-test between base chart and reference chart
        pvals_for_base_chart.append(pval_with_ref_chart)
    Pvals.append(pvals_for_base_chart)
    
for i in range(len(Pvals)):
    print (Pvals[i])
        