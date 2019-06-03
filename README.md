# Tensorflow Neural Network
A basic implementation of an artificial neural network on python using numpy, pandas, matplotlib and tensorflow.

The dataset used for this project is the "Boston Housing" dataset. Information about this dataset is available on [Kaggle](https://www.kaggle.com/c/boston-housing/overview/description).
The main goal of this neural network is to predict some housing prices based on prior data.

## The problem
This is a regression problem, for this we will be using a 4-layer basic neural network which, as we can see on the following plots, gets
easily overfitted. For this reason we will use a training cycle consisting on 100 epochs.

![100-epoch](https://github.com/maccarini/tensorflow-neural-net/blob/master/100epoch.png "100-Epoch example")
![300-epoch](https://github.com/maccarini/tensorflow-neural-net/blob/master/300epoch.png "300-Epoch example")
