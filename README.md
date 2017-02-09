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

##Requirements
* Python 3.6
* numpy
* scipy
* matplotlib
* sklearn
* Theano
* Keras

##Installation
I recommend using Anaconda to manage packages and Python versions. You can find
the installer for Anaconda [here](https://www.continuum.io/downloads). 

* Create Anaconda environment
  * `conda create -n yourenvname python=3 anaconda`
  * This will initialize a conda environment with all of the default packages
    many of which are needed for this project.
* Activate the conda environment
  * `source activate yourenvname`
* Install additional packages
  * Unfortunately installing Theano and Keras through the `conda install`
    command does not work.
  * `pip install Theano`
  * `pip install Keras`
* Deactivate your conda environment
  * `deactivate yourenvname`
