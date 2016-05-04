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

from pybrain.tools.shortcuts import LinearLayer, SigmoidLayer, FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.optimization import CMAES
from pybrain.structure import FeedForwardNetwork

# def MakeDataSet():
    #use sinx as data

def MakeNeuralNet():
    net = FeedForwardNetwork()
    inLayer = LinearLayer(3, name='Jon')
    hiddenLayer = SigmoidLayer(2, name='Ryan')
    outLayer = LinearLayer(1, name='Sam')

    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    net.addInputModule(inLayer)
    net.addModule(hiddenLayer)
    net.addOutputModule(outLayer)
    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)

    net.sortModules()
    ds = SupervisedDataSet(2, 1)
    trainer = BackpropTrainer(net, ds) #trains for one epoch
    #trainer.trainUntilConvergence trains to a specific error
    print net

# net.activate([1, 2])

#
# for i in range(len(X)):
#     ds.addSample(X[i], Y[i])
#
# def objF(x):
#     return sum(x**2)
#
# x0 = ([2.1, -1])
# l = CMAES(objF, x0)
# l.maxEvaluations = 200000000000

MakeNeuralNet()
