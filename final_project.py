from pybrain.tools.shortcuts import LinearLayer, SigmoidLayer, FullConnection
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.optimization import CMAES
from pybrain.structure import FeedForwardNetwork

# def MakeDataSet():
    #use sinx as data
class NeuralNetwork():
    def __init__(self):
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

    def objF(self, x):
        return sum(x**2)

    def TrainNeuralNet(self):
        x0 = ([2.1, -1])
        l = CMAES(objF, x0)
        l.maxEvaluations = 200
        trainer = BackpropTrainer(net, ds)



# net.activate([1, 2])

# for i in range(len(X)):
#     ds.addSample(X[i], Y[i])

net = NeuralNetwork()
print net.TrainNeuralNet()
