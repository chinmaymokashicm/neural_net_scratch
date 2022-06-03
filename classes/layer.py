"""Layer class
"""
import numpy as np
from .neuron import Neuron

class Layer:
    def __init__(self, nodes_previous_layer, nodes=5) -> None:
        self.__n_prev = nodes_previous_layer
        self.__nodes = nodes
        # Construct the layer
        self.__layer = [Neuron(n_previous_layers=self.__n_prev) for _ in range(self.__nodes)]

    @property
    def n_prev(self) -> int:
        """Getter for n_prev
        """
        return self.__n_prev

    @property
    def nodes(self) -> int:
        """Getter for nodes
        """
        return self.__nodes

    @property
    def layer(self) -> list:
        """Getter for layer
        """
        return self.__layer

    @n_prev.setter
    def n_prev(self, n_prev) -> None:
        """Setter for n_prev
        """
        self.__n_prev = n_prev
    
    @nodes.setter
    def nodes(self, nodes) -> None:
        """Setter for nodes
        """
        self.__nodes = nodes
    
    @layer.setter
    def layer(self, layer) -> None:
        """Setter for layer
        """
        self.__layer = layer
    
    