from experiment import *
from lenet import get_image_patches
from data_helper import load_data
from kmeans import k_means

datasets = load_data('mnist.pkl', notShared=True)
trainSetX, trainSetY = datasets[0]

filterShape = (5,5)
stride = (1,1)
inputShape = (28,28)
nkerns = (6, 16)

trainSetX = trainSetX[0:1]
patchVecs = []
total = len(trainSetX)
for i, trainX in enumerate(trainSetX):
    if i % (total / 50) == 0:
        print '%.2f%%' % ((float(i) / float(total)) * 100.)
    patches = get_image_patches(trainX, inputShape, stride, filterShape)
    print len(patches)
    for patch in patches:
        # Flatten out the patch
        flattened = [item for sublist in patch for item in sublist]
        patches.append(flattened)

centroids = k_means(patchVecs, nkerns[0])
