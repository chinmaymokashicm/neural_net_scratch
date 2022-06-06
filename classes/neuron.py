"""Neuron class
"""
import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, n_previous_layers=3) -> None:
        """Neuron class constructor
        """
        if(n_previous_layers is not None):
            self.__W = np.random.normal(0, 1, n_previous_layers)
            self.__W = np.arange(0, n_previous_layers*10, 10)
            self.__b = -2
            self.__A = 0
        else: # If the neuron is in the first layer
            self.__W = None
            self.__b = None
            self.__A = None

    def forward_prop(self, X, activation="relu") -> None:
        """Forward propagation.

        Args:
            X (np.ndarray): Input
            activation (str, optional): Activation function. Defaults to "relu".
        """
        if(self.__W is not None):
            a = X @ self.__W.T + self.__b
            dict_activation = {
                "relu": lambda a: np.maximum(0, a)
            }
            if(activation not in dict_activation):
                activation = "relu"
            self.__A = dict_activation[activation](a)
        else:
            self.__A = X # In this case, X would be an integer

    @property
    def W(self) -> np.ndarray:
        """Getter for W
        """
        return self.__W

    @property
    def b(self) -> np.ndarray:
        """Getter for b
        """
        return self.__b
    
    @property
    def A(self) -> np.ndarray:
        """Getter for A
        """
        return self.__A
    
    @property
    def output(self) -> np.ndarray:
        """Getter for output
        """
        return self.__A

    @W.setter
    def W(self, W) -> None:
        """Setter for W
        """
        self.__W = W

    @b.setter
    def b(self, b) -> None:
        """Setter for b
        """
        self.__b = b

    @A.setter
    def A(self, A) -> None:
        """Setter for A
        """
        self.__A = A

    @output.setter
    def output(self, output) -> None:
        """Setter for output
        """
        self.__A = output
