import numpy as np
from scipy import stats
from pprint import pprint
from MNIST import MNIST

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Dense

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

model = Sequential()
model.add(Dense(500, input_shape=[784,], activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

mnist = MNIST()
xtrain, ytrain, xtest, ytest = mnist.load(threeDim=False)

# model.fit(xtrain, ytrain, epochs=10, batch_size=64) # Untrained model
model.load_weights("model.h5") # Loading pre-trained model
stats = model.evaluate(xtest, ytest, batch_size=64)
y_hat = model.predict(xtest)
accuracy = stats[1] * 100

print (accuracy)

for kval in range(1, 21):
    kNN = KNeighborsClassifier(n_neighbors=kval)
    print ('For kNN {}'.format(kval))
    trials_X, trials_y = mnist.permute_set(xtrain, ytrain, xtest, ytest, 5)
    print ()