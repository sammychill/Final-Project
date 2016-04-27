import numpy as np
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.optimization import CMAES


net = buildNetwork(2, 3, 1)

result = net.activate([2, 1])

ds = SupervisedDataSet(2, 1)

X = np.array([[3, 5], [5, 1], [10, 2]])
Y = np.array([[75], [82], [93]])


for i in range(len(X)):
    ds.addSample(X[i], Y[i])

def objF(x):
    return sum(x**2)

x0 = ([2.1, -1])
l = CMAES(objF, x0)
l.maxEvaluations = 200

trainer = BackpropTrainer(net, ds) #trains for one epoch
#trainer.trainUntilConvergence trains to a specific error
l.learn()

print trainer.trainUntilConvergence()
