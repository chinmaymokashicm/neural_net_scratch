from classes.neuron import Neuron
from classes.layer import Layer
from classes.network import Network
from classes.loss import Loss

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

X, y = spiral_data(100, 3)

np.printoptions(precision=4)

network = Network(X, y, hidden=[2, 3, 2])
# output = network.forward_prop(X[0])
network.pprint()
loss_vector = network.train(batch_size=30)
network.pprint()
print(network.get_schema())

# print(output)

# print(Loss().cross_entropy(y_actual=[1, 0, 0, 0], y_pred=[0.8, 0.1, 0.01, 0.09]))

# network.train(X, y, batch_size=25)