from classes.neuron import Neuron
from classes.layer import Layer
from classes.network import Network
import numpy as np

X = np.array([[1,2,3], [0,5,10]])
neuron = Neuron(n_previous_layers=X.shape[1])

# neuron.forward_prop(X)
# print(neuron.A)

layer = Layer(nodes_previous_layer=3, nodes=5)
layer.layer[0].forward_prop(X)
# print(layer.layer[0].A)

network = Network(n_inputs=5, n_hidden=3, n_outputs=4, hidden=[2, 3, 2])
network.forward_prop([1, 2, 3, 4, 5])
