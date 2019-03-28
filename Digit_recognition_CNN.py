#import dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras 
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.layers import Convolution2D
from keras.utils import np_utils

#download dataset
(X_train, y_train) , (X_test, y_test) = mnist.load_data()
#Pre-processing data----------------------------------------------------------------------------
#reshape input vectors, convert to float and normalize them to values between 0 to 1
X_train = X_train.reshape(60000,28,28,1).astype('float32')/255
X_test = X_test.reshape(10000,28,28,1).astype('float32')/255

#create label matrix for label eg: [0001000000] for y = 3
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#number of classes for output layer
classes = y_train.shape[1]

#making the model-------------------------------------------------------------------------------
#making the convnet
def CNN():
  model = Sequential()
  model.add(Convolution2D(30, (5,5), input_shape = (28,28,1), activation='relu'))
  model.add(MaxPooling2D( pool_size=(2,2) ))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(125, activation = 'relu'))
  model.add(Dense(classes, activation = 'softmax'))
  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model

#create an object of the function
model = CNN()

#train the model on downloaded dataset and validate on X_test and y_test-----------------------
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 5, batch_size = 200, verbose = 2)

#evaluate our trained model on test data and print accuracy metrics
score = model.evaluate(X_test, y_test)



