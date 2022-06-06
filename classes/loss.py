"""Class for loss calculations
"""
import numpy as np

class Loss:
    def __init__(self) -> None:
        pass

    def cross_entropy(self, y_actual, y_pred):
        # Cross-entropy is a measure of the difference between 
        # two probability distributions for a given random variable or set of events.
        # https://machinelearningmastery.com/cross-entropy-for-machine-learning/
        return -np.sum(np.multiply(y_actual, list(map(np.log, y_pred))))

    def cost_function(self):
        pass