"""Neuron class
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Neuron:
    def __init__(self, n_previous_layers=3) -> None:
        """Neuron class constructor
        """
        self.__W = np.random.normal(0, 1, n_previous_layers)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X) -> None:
        """Forward propagation
        """
        self.__A = np.dot(X, self.__W) + self.__b

    def evaluate(self, X) -> None:
        """Evaluate the neuron
        """
        self.forward_prop(X)
        self.__A = 1 / (1 + np.exp(-self.__A))
    
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

    def cost(self, Y, A) -> float:
        """Cost function
        """
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
    
    def evaluate_cost(self, X, Y) -> float:
        """Evaluate cost function
        """
        self.evaluate(X)
        return self.cost(Y, self.__A)

    def gradient_descent(self, X, Y, A, alpha=0.05) -> None:
        """Gradient descent
        """
        dZ = A - Y
        m = X.shape[1]
        self.__W = self.__W - (alpha / m) * np.dot(X.T, dZ)
        self.__b = self.__b - (alpha / m) * dZ.sum()

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100) -> None:
        """Train the neuron
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        _iterations = 0
        if verbose is True:
            print("Cost after {} iterations: {}".format(_iterations, self.evaluate_cost(X, Y)))
        if graph is True:
            cost = []
        while _iterations < iterations:
            self.evaluate(X)
            if verbose is True:
                if _iterations % step == 0 or _iterations == iterations:
                    print("Cost after {} iterations: {}".format(_iterations, self.evaluate_cost(X, Y)))
            if graph is True:
                cost.append(self.evaluate_cost(X, Y))
            self.gradient_descent(X, Y, self.__A, alpha)
            _iterations += 1
        if verbose is True:
            print("Cost after {} iterations: {}".format(_iterations, self.evaluate_cost(X, Y)))
        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), cost, 'r')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
    
    def predict(self, X) -> np.ndarray:
        """Predict
        """
        self.evaluate(X)
        return np.where(self.__A >= 0.5, 1, 0)

    def save(self, filename: str) -> None:
        """Save the instance object to a file in pickle format
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename: str) -> None:
        """Load a pickled file
        """
        try:
            with open(filename, 'rb') as f:
                self = pickle.load(f)
        except FileNotFoundError:
            pass
    
    def __str__(self) -> str:
        """String representation
        """
        return "Neuron with {} nodes".format(self.W.shape[0])
    
    def __repr__(self) -> str:
        """String representation
        """
        return self.__str__()
    
    def __del__(self) -> None:
        """Delete the instance
        """
        print("Neuron instance deleted")
    
    def __add__(self, other) -> None:
        """Addition
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be added with a Neuron instance")
        return Neuron(self.W + other.W)
    
    def __sub__(self, other) -> None:
        """Subtraction
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be subtracted with a Neuron instance")
        return Neuron(self.W - other.W)

    def __mul__(self, other) -> None:
        """Multiplication
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be multiplied with a Neuron instance")
        return Neuron(self.W * other.W)

    def __truediv__(self, other) -> None:
        """Division
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be divided with a Neuron instance")
        return Neuron(self.W / other.W)

    def __pow__(self, other) -> None:
        """Power
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be powered with a Neuron instance")
        return Neuron(self.W ** other.W)

    def __radd__(self, other) -> None:
        """Addition
        """
        return self.__add__(other)

    def __rsub__(self, other) -> None:
        """Subtraction
        """
        return self.__sub__(other)

    def __rmul__(self, other) -> None:
        """Multiplication
        """
        return self.__mul__(other)

    def __rtruediv__(self, other) -> None:
        """Division
        """
        return self.__truediv__(other)

    def __rpow__(self, other) -> None:
        """Power
        """
        return self.__pow__(other)
    
    def __eq__(self, other) -> bool:
        """Equality
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be compared with a Neuron instance")
        return self.W == other.W

    def __ne__(self, other) -> bool:
        """Inequality
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be compared with a Neuron instance")
        return self.W != other.W

    def __lt__(self, other) -> bool:
        """Less than
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be compared with a Neuron instance")
        return self.W < other.W

    def __le__(self, other) -> bool:
        """Less than or equal to
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be compared with a Neuron instance")
        return self.W <= other.W

    def __gt__(self, other) -> bool:
        """Greater than
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be compared with a Neuron instance")
        return self.W > other.W
    
    def __ge__(self, other) -> bool:
        """Greater than or equal to
        """
        if type(other) is not Neuron:
            raise TypeError("Neuron must be compared with a Neuron instance")
        return self.W >= other.W

    def __len__(self) -> int:
        """Length
        """
        return self.W.shape[0]

    def __getitem__(self, key) -> np.ndarray:
        """Indexing
        """
        return self.W[key]

    def __setitem__(self, key, value) -> None:
        """Indexing
        """
        self.W[key] = value

    def __iter__(self) -> None:
        """Iteration
        """
        return iter(self.W)

    def __contains__(self, item) -> bool:
        """Containment
        """
        return item in self.W

    def __call__(self, X) -> np.ndarray:
        """Call
        """
        return self.evaluate(X)

    def __getstate__(self) -> dict:
        """Get state
        """
        return {'W': self.W}

    def __setstate__(self, state: dict) -> None:
        """Set state
        """
        self.W = state['W']
    
    def __getattr__(self, name: str) -> None:
        """Get attribute
        """
        if name == 'W':
            return self.__dict__[name]
        else:
            raise AttributeError("Attribute {} not found".format(name))

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute
        """
        if name == 'W':
            self.__dict__[name] = value
        else:
            raise AttributeError("Attribute {} not found".format(name))

    def __delattr__(self, name: str) -> None:
        """Delete attribute
        """
        if name == 'W':
            del self.__dict__[name]
        else:
            raise AttributeError("Attribute {} not found".format(name))

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the neuron
        """
        return self.W @ X