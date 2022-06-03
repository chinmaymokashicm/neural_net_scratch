"""Layer class
"""
import numpy as np

class Layer:
    """Layer class
    """
    def __init__(self, nodes):
        """Constructor
        """
        self.nodes = nodes
        self.weights = []
        self.biases = []
        self.zs = []
        self.as = []

    def set_weights(self, weights):
        """Set weights
        """
        self.weights = weights

    def set_biases(self, biases):
        """Set biases
        """
        self.biases = biases

    def forward_prop(self, X, activation="sigmoid"):
        """Forward propagation
        """
        self.zs = []
        self.as = []
        for i in range(len(self.weights)):
            self.zs.append(np.dot(self.weights[i], X) + self.biases[i])
            if activation == "sigmoid":
                self.as.append(sigmoid(self.zs[i]))
            elif activation == "tanh":
                self.as.append(tanh(self.zs[i]))
            elif activation == "relu":
                self.as.append(relu(self.zs[i]))
            elif activation == "leaky_relu":
                self.as.append(leaky_relu(self.zs[i]))
            elif activation == "softmax":
                self.as.append(softmax(self.zs[i]))
            else:
                raise ValueError("activation must be 'sigmoid', 'tanh', 'relu', 'leaky_relu' or 'softmax'")
        return self.as[-1]

    def back_prop(self, X, Y, A, activation="sigmoid"):
        """Back propagation
        """
        dZ = []
        dW = []
        db = []
        if activation == "sigmoid":
            dZ.append(A - Y)
            dW.append(np.dot(dZ[-1], X.T))
            db.append(np.sum(dZ[-1], axis=1, keepdims=True))
        elif activation == "tanh":
            dZ.append(A - Y)
            dW.append(np.dot(dZ[-1], X.T))
            db.append(np.sum(dZ[-1], axis=1, keepdims=True))
        
        return dW, db

    def update(self, dW, db, learning_rate):
        """Update weights and biases
        """
        self.weights = [w - learning_rate * dW[i] for i, w in enumerate(self.weights)]
        self.biases = [b - learning_rate * db[i] for i, b in enumerate(self.biases)]
    
    def cost(self, Y, A):
        """Cost function
        """
        return -np.sum(Y * np.log(A)) / Y.shape[1]

    def evaluate_cost(self, X, Y, activation="sigmoid"):
        """Evaluate cost function
        """
        self.forward_prop(X, activation)
        return self.cost(Y, self.as[-1])
    
    def evaluate(self, X, Y, activation="sigmoid"):
        """Evaluate the neuron
        """
        self.forward_prop(X, activation)
        return self.as[-1]
    
    def evaluate_accuracy(self, X, Y, activation="sigmoid"):
        """Evaluate the accuracy
        """
        self.forward_prop(X, activation)
        return np.sum(np.argmax(self.as[-1], axis=0) == np.argmax(Y, axis=0)) / Y.shape[1]
    
    def train(self, X, Y, activation="sigmoid", learning_rate=0.01, epochs=100):
        """Train the neuron
        """
        for i in range(epochs):
            self.forward_prop(X, activation)
            dW, db = self.back_prop(X, Y, self.as[-1], activation)
            self.update(dW, db, learning_rate)
            print("Epoch: {}, Cost: {}".format(i, self.evaluate_cost(X, Y, activation)))

    def predict(self, X, activation="sigmoid"):
        """Predict the output
        """
        return self.evaluate(X, activation)

    def predict_accuracy(self, X, Y, activation="sigmoid"):
        """Predict the accuracy
        """
        return self.evaluate_accuracy(X, Y, activation)
    
    def predict_class(self, X, activation="sigmoid"):
        """Predict the class
        """
        return np.argmax(self.predict(X, activation), axis=0)

    def predict_probability(self, X, activation="sigmoid"):
        """Predict the probability
        """
        return self.predict(X, activation)

    def predict_probability_class(self, X, activation="sigmoid"):
        """Predict the probability class
        """
        return np.argmax(self.predict_probability(X, activation), axis=0)

    def predict_probability_class_one_hot(self, X, activation="sigmoid"):
        """Predict the probability class one hot
        """
        return self.predict(X, activation)
    
    def predict_class_one_hot(self, X, activation="sigmoid"):
        """Predict the class one hot
        """
        return np.argmax(self.predict(X, activation), axis=0)

    def predict_class_one_hot_probability(self, X, activation="sigmoid"):
        """Predict the class one hot probability
        """
        return self.predict(X, activation)

    def predict_class_one_hot_probability_class(self, X, activation="sigmoid"):
        """Predict the class one hot probability class
        """
        return np.argmax(self.predict_class_one_hot_probability(X, activation), axis=0)
    
    def predict_class_one_hot_probability_class_one_hot(self, X, activation="sigmoid"):
        """Predict the class one hot probability class one hot
        """
        return self.predict(X, activation)

    def predict_class_one_hot_probability_class_one_hot_probability(self, X, activation="sigmoid"):
        """Predict the class one hot probability class one hot probability
        """
        return self.predict(X, activation)

    