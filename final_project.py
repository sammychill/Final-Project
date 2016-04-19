# import numpy as np
#
# X = np.array([3, 5], [5, 1], [10, 2])
# Y = np.array([75], [82], [93])
# y = Y/10.
#
# class Neural_Network(object):
#     def __init__(self):
#         '''
#         Hyperparameters
#         constants that describe the structure/behavior of network
#         never updated
#         '''
#         self.inputLayerSize = 2
#         self.hiddenLayerSize = 3
#         self.outputLayerSize = 1
#         '''
#         Weights
#         randomly selected from standard deviation curve (from neg
#         infinity to pos infinity)
#         returns array sized according to arguments given
#         '''
#         self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
#         self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
#     def forward(self):
#         '''
#         passes data through network using sigmoid function
#         returns yHat
#         '''
#         self.z2 = np.dot(X, self.W1) #multiplies matrices
#         self.a2 = self.sigmoid(self.z2) #uses activation function
#         self.z3 = np.dot(self.a2, self.W2) #multiplies matrices
#         yHat = self.sigmoid(self.z3) #uses activation function
#         return yHat
#
#     def sigmoid(self, z):
#         return 1/(1+np.exp(-z))
#
# network = Neural_Network()
# yHat = network.forward(X)
# print yHat, y
#learn how to use open CV, take his data

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.optimization import CMAES


net = buildNetwork(2, 3, 1)

result = net.activate([2, 1])

ds = SupervisedDataSet(2, 1)

X = ([3, 5], [5, 1], [10, 2])
Y = ([75], [82], [93])

for i in range(len(X)):
    ds.addSample(X[i], Y[i])

def objF(x):
    return sum(x**2)

x0 = ([2.1, -1])
l = CMAES(objF, x0)
l.maxEvaluations = 200000000000

trainer = BackpropTrainer(net, ds) #trains for one epoch
#trainer.trainUntilConvergence trains to a specific error
l.learn()

print trainer.trainUntilConvergence()
