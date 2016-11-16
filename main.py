from tester import add_convlayer
from tester import add_fclayer
from tester import fetch_data
from sklearn.cluster import MiniBatchKMeans

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD
from keras.utils import np_utils

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import datasets
from clustering import kmeans

import csv
import pickle
import warnings
import os


(train_data, test_data, train_labels, test_labels) = fetch_data(0.3)

train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

use_amount = 20000
train_data = np.array(train_data[0:use_amount])
train_labels = np.array(train_labels[0:use_amount])
test_data = np.array(test_data[0:use_amount])
test_labels = np.array(test_labels[0:use_amount])

# Only use a given amount of the training data.
split = use_amount / 2
train_data_update = train_data[0:split]
train_labels_update = train_labels[0:split]
train_data_cluster = train_data[split:]

train_labels_count = len(train_labels[0])

train_data_cluster = np.array(train_data_cluster)
sp = train_data_cluster.shape
train_data_cluster = train_data_cluster.reshape(sp[0], sp[1] * sp[2] * sp[3])

# Determine labels for unsupervised learning.
# centers, auto_labels = kmeans(train_data_cluster, train_labels_count, 128, metric='cosine')

# Determine labels for unsupervised learning.
mbk = MiniBatchKMeans(init='k-means++',
                        n_clusters=train_labels_count,
                        batch_size=128,
                        max_no_improvement=10,
                        reassignment_ratio=0.01,
                        verbose=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    auto_labels = mbk.fit_predict(train_data_cluster)

# Check how 'off' the label is.
wrong_count = 0
for i, auto_label in enumerate(auto_labels):
    real_label = train_labels_update[i]
    # Convert to the index.
    for i, val in enumerate(real_label):
        if val == 1:
            real_label = i
            break
    if real_label != auto_label:
        wrong_count += 1


print (float(wrong_count) / float(len(auto_labels)))
raise ValueError()

input_shape = (1, 28, 28)
subsample=(1,1)
filter_size=(5,5)
batch_size = 5
nkerns = (6, 16)

# Train the model using these labels and data.

model = Sequential()

convout0_f = add_convlayer(model, nkerns[0], subsample, filter_size, input_shape=input_shape)

convout1_f = add_convlayer(model, nkerns[1], subsample, filter_size)

model.add(Flatten())

fc0_f = add_fclayer(model, 120)

fc1_f = add_fclayer(model, 84)

classification_layer = Dense(10)
model.add(classification_layer)

model.add(Activation('softmax'))

opt = SGD(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

train_data_cluster = train_data_cluster.reshape(sp[0], sp[1], sp[2], sp[3])
# Encode the labels to a one hot representation.
encoded_labels = []
for auto_label in auto_labels:
    encoded_label = [0] * 10
    encoded_label[auto_label] = 1
    encoded_labels.append(encoded_label)

encoded_labels = np.array(encoded_labels)

model.fit(train_data_cluster, encoded_labels, batch_size=batch_size, nb_epoch=20, verbose = 1)

(loss, accuracy) = model.evaluate(test_data, test_labels, batch_size = batch_size, verbose = 1)
print ''
print 'Accuracy %.9f%%' % (accuracy * 100.)
