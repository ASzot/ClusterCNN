# ClusterCNN

## Summary
Use k-means clustering from the input data set to partially train the network by obtain weights for the convolutional filters.
Then compare the impact stochastic gradient descent has on this already partially trained network.


## Motivation
Training Convolutional Neural Networks (CNNs) has typically taken vast amounts of labeled data.
Obtaining this labeled data is extremely difficult often requiring countless hours of manual annotation.
This experiment serves to explore if clustering techniques can be used to automatically set the weights of a CNN.
The motivation behind this experiment is viewing convolution filter weights as anchor vectors as described in
[this](https://arxiv.org/abs/1609.04112) paper.
