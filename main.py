from tester import add_convlayer
from tester import add_fclayer
from sklearn.cluster import MiniBatchKMeans



(train_data, test_data, train_labels, test_labels) = fetch_data(0.3)

remaining = int(len(train_data) * train_percentage)

train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

use_amount = 10000
train_data = np.array(train_data[0:use_amount])
train_labels = np.array(train_labels[0:use_amount])
test_data = np.array(test_data[0:use_amount])
test_labels = np.array(test_labels[0:use_amount])

# Only use a given amount of the training data.
split = 5000
train_data_update = train_data[0:split]
train_labels_update = train_labels[0:split]
train_data_cluster = train_data[split:]

train_labels_count = len(train_labels[0])

# Determine labels for unsupervised learning.
mbk = MiniBatchKMeans(init='k-means++',
                        n_clusters=train_labels_count,
                        batch_size=batch_size,
                        max_no_improvement=10,
                        reassignment_ratio=0.01,
                        verbose=False)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    auto_labels = mbk.fit_predict(train_data_cluster)

print auto_labels[0]

input_shape = (1, 28, 28)
subsample=(1,1)
filter_size=(5,5)
batch_size = 5
nkerns = (6, 16)


model = Sequential()

convout0_f = add_convlayer(model, nkerns[0], subsample, filter_size, input_shape=input_shape, weights=input_centroids[0])

convout1_f = add_convlayer(model, nkerns[1], subsample, filter_size, input_shape=input_shape, weights = input_centroids[1])

model.add(Flatten())

fc0_f = add_fclayer(model, 120, weights = input_centroids[2])

fc1_f = add_fclayer(model, 84, weights=input_centroids[3])

classification_layer = Dense(10)
model.add(classification_layer)

model.add(Activation('softmax'))

print 'Compiling model'
opt = SGD(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(scaled_train_data, train_labels, batch_size=batch_size, nb_epoch=20, verbose = 1)

(loss, accuracy) = model.evaluate(test_data, test_labels, batch_size = batch_size, verbose = 1)
print ''
print 'Accuracy %.9f%%' % (accuracy * 100.)
