"""Neural Network class
"""
import numpy as np
from .layer import Layer

class Network:
    def __init__(self, n_inputs, n_hidden, n_outputs, hidden=None) -> None:
        # Construct the network
        self.__network = []
        # First layer
        self.__network.append(Layer(nodes_previous_layer=None, nodes=n_inputs))
        # Hidden layers
        for i in range(n_hidden):
            self.__network.append(Layer(nodes_previous_layer=n_inputs if i == 0 else hidden[i - 1], nodes=hidden[i]))
        # Last (output) layer
        self.__network.append(Layer(nodes_previous_layer=hidden[-1] if hidden is not None else n_inputs, nodes=n_outputs))
        print(self.pprint())
        
    def pprint(self):
        for layer_number, layer in enumerate(self.__network):
            print(f"Layer number: {layer_number + 1} Number of nodes: {layer.nodes}")
            for neuron_number, neuron in enumerate(layer.layer):
                print(neuron.A)