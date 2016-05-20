'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25 test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility, same random each time
#1337 is starting point

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot
import matplotlib as mpl

#this is where we groom our data and split it up

batch_size = 128 #too small (overfitting), too large (underfitting)
#checked by validation dataset to find optimal batch_size for memory/speed/accuracy
#determines number of iterations per epoch
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#numpy, shape()
#numpy, reshape(a, new_shape, order), resizes matrix
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32') #numpy
X_test = X_test.astype('float32')
X_train /= 255 #divides X_train by 255 and sets X_train equal
#X_train = X_train/255 = 235
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
#this is where we build our NN

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten()) #multiplies dimensions by each other to get one number
model.add(Dense(128)) #specifies that output matrix has 128 columns
model.add(Activation('relu'))
model.add(Dropout(0.5)) #stops overfitting by dropping half of inputs?
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

'''
model.compile is method from Sequential class in keras database
prints:
Epoch 1/12
60000/60000 [==============================]
ETA: 162s - loss: 0.3675 - acc: 0.8887 - val_loss: 0.0935 - val_acc: 0.9728

categorical_crossentropy -- gradient descent, cost function that alters
weights and bias according to errors

optimizer='sgd' -- shorthand for "from keras import SGD", template, import
separately if we want to modify, stochastic gradient descent
'''
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta', #addadelta
              metrics=['accuracy'])

#populate network with data, fit w/ training set, verify w/ validation set
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
