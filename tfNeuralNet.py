#Basic implementation of an ANN using tensorflow

#Importing the Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('housing.data', sep ='\s+', header=None)
X = data.iloc[:, 0:13].values
y = data.iloc[:,13:14].values

# Splitting train and test sets
Xtrain, ytrain = np.float32(X[:404]), np.float32(y[:404])
Xtest, ytest = np.float32(X[404:]), np.float32(y[404:])

# Scaling data using min-max normalization
from sklearn.preprocessing import MinMaxScaler
xscaler = MinMaxScaler()
yscaler = MinMaxScaler()

Xtrain = xscaler.fit_transform(Xtrain)
Xtest = xscaler.transform(Xtest)

ytrain = yscaler.fit_transform(ytrain)
ytest = yscaler.transform(ytest)

# Building the architecture of the net
inputLayerSize = 13
hiddenLayer1Size = 7
hiddenLayer2Size = 4
outputLayerSize = 1

# Weights and biases initialization
w1 = tf.Variable(tf.random_uniform([inputLayerSize, hiddenLayer1Size], dtype='float32'))
w2 = tf.Variable(tf.random_uniform([hiddenLayer1Size, hiddenLayer2Size], dtype='float32'))
w3 = tf.Variable(tf.random_uniform([hiddenLayer2Size, outputLayerSize], dtype='float32'))
b1 = tf.Variable(tf.zeros([hiddenLayer1Size], dtype='float32'))
b2 = tf.Variable(tf.zeros([hiddenLayer2Size], dtype='float32'))
b3 = tf.Variable(tf.zeros([outputLayerSize], dtype='float32'))

# Feedforward function
def feedForward(Xtrain):
  z2 = tf.add(tf.matmul(Xtrain,w1), b1)
  a2 = tf.nn.relu(z2)
  
  z3 = tf.add(tf.matmul(a2,w2), b2)
  a3 = tf.nn.relu(z3)
  
  y_hat = tf.add(tf.matmul(a3,w3), b3)
  
  return y_hat

# Initializing placeholders to be replaced with X and y values on training
xx = tf.placeholder("float", )
yy = tf.placeholder("float")

# Output and cost functions
yhat = feedForward(xx)
cost = tf.reduce_mean(tf.square(yhat-yy))

# Using Adam optimizer for backprop with a learning rate of 0.001
train = tf.train.AdamOptimizer(0.001).minimize(cost)

# Empty lists to be filled with cost values
train_cost = []
test_cost = []
n_epochs = 100

# Training loop
with tf.Session() as sess:
  
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  
  for i in range (n_epochs):
    
    for j in range(Xtrain.shape[0]):
      sess.run([cost, train], feed_dict={xx:Xtrain[j,:].reshape(1,inputLayerSize), yy:ytrain[j]})
      
    train_cost.append(sess.run(cost, feed_dict={xx:Xtrain,yy:ytrain}))
    test_cost.append(sess.run(cost, feed_dict={xx:Xtest,yy:ytest}))
    print('Epoch:',i,'Cost:', train_cost[i], 'Val Cost:', test_cost[i])
    
  ypred = sess.run(yhat, feed_dict={xx:Xtest})
  
# Visualizing training and validation cost curves
xaxis = np.linspace(0, n_epochs, n_epochs)
plt.plot(xaxis, train_cost, label='Train cost')
plt.plot(xaxis, test_cost, label='Test Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend()
plt.show()

# Clear plot
#plt.clf()