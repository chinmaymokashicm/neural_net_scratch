"""Neural Network class
"""
import numpy as np
from .layer import Layer

class Network:
    def __init__(self, n_inputs, n_hidden, n_outputs, hidden=None) -> None:
        self.__n_inputs = n_inputs
        # Construct the network
        self.__network = []
        # First layer
        self.__network.append(Layer(nodes_previous_layer=None, nodes=n_inputs))
        # Hidden layers
        for i in range(n_hidden):
            self.__network.append(Layer(nodes_previous_layer=n_inputs if i == 0 else hidden[i - 1], nodes=hidden[i]))
        # Last (output) layer
        self.__network.append(Layer(nodes_previous_layer=hidden[-1] if hidden is not None else n_inputs, nodes=n_outputs))

    def forward_prop(self, X):
        if(len(X) != self.__n_inputs):
            raise Exception(f"Incorrect input size. Correct size is {self.__n_inputs} but provided input size is {len(X)}")
        
        X_current = np.copy(X)
        for neuron_number in range(len(self.__network[0].layer)):
            self.__network[0].layer[neuron_number].forward_prop(X_current[neuron_number])
            X_current[neuron_number] = self.__network[0].layer[neuron_number].A
        
        for layer_number, layer in self.__network[1:]:
            
            pass

        self.pprint()
        
    def pprint(self):
        for layer_number, layer in enumerate(self.__network):
            print(f"Layer number: {layer_number + 1} Number of nodes: {layer.nodes}")
            for neuron_number, neuron in enumerate(layer.layer):
                print(f"Neuron number:{neuron_number + 1} Activation: {neuron.A}")