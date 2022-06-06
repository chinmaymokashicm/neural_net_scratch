"""Neural Network class
"""
import numpy as np
from tqdm import tqdm
from flask_restful import Resource
import json

from .loss import Loss
from .layer import Layer

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Network:
    def __init__(self, X, y, hidden=None) -> None:
        try:
            # See if the input and output arrays have consistent sizes
            X = np.array(X)
            y = np.array(y)
        except Exception as e:
            print("Check input or output arrays!")
            print(e)
            return
        self.n_inputs = X.shape[1]
        self.n_outputs = len(np.unique(y))
        self.n_hidden = len(hidden)
        self.X = X
        self.y = self.one_hot_encode(y)
        self.loss_obj = Loss()
        # Construct the network
        self.__network = []
        # First layer
        self.__network.append(Layer(nodes_previous_layer=None, nodes=self.n_inputs))
        # Hidden layers
        for i in range(len(hidden)):
            self.__network.append(Layer(nodes_previous_layer=self.n_inputs if i == 0 else hidden[i - 1], nodes=hidden[i]))
        # Last (output) layer
        self.__network.append(Layer(nodes_previous_layer=hidden[-1] if hidden is not None else self.n_inputs, nodes=self.n_outputs))
        self.__output = []

    def one_hot_encode(self, y):
        return np.eye(self.n_outputs)[y]

    def softmax_activation(self, array_activation):
        array_activation = [activation - max(array_activation) for activation in array_activation] # To avoid overflow after exponentiation
        exp_a = [pow(np.e, activation) for activation in array_activation]
        sum_exp_a = np.sum(exp_a)
        exp_a = [exp_activation/sum_exp_a for exp_activation in exp_a]
        return exp_a

    def forward_prop(self, sample): 
        X_current = np.copy(sample)
        for neuron_number in range(len(self.__network[0].layer)):
            self.__network[0].layer[neuron_number].forward_prop(X_current[neuron_number])
            X_current[neuron_number] = self.__network[0].layer[neuron_number].A
        
        for layer_number in range(len(self.__network)):
            if(layer_number == 0):
                continue
            X_new = []
            for neuron_number in range(len(self.__network[layer_number].layer)):
                self.__network[layer_number].layer[neuron_number].forward_prop(X_current)
                X_new.append(self.__network[layer_number].layer[neuron_number].A)
            X_current = np.array(X_new)
        self.__output = self.softmax_activation([neuron.A for neuron in self.__network[-1].layer])
        return self.__output

    def back_prop(self):
        pass

    def pipeline_per_sample(self, sample, output):
        # Steps taken for every individual sample
        # Get prediction
        prediction = self.forward_prop(sample)
        # Compare with actual output, get loss

        loss = self.loss_obj.cross_entropy(output, prediction)
        return loss

    def train(self, batch_size=100, n_iters=100):
        # Batch forward propagation
        list_batches_X = [self.X[i: i+batch_size] for i in range(0, len(self.X), batch_size)]
        list_batches_y = [self.y[i: i+batch_size] for i in range(0, len(self.y), batch_size)]
        loss_vector_history = []
        for epoch in range(0, n_iters):
            loss_vector_batch = []
            for batch_number in tqdm(range(len(list_batches_X))):
                """
                Store activations for every batch (at layer level)
                
                """
                # Generate a loss vector for every batch
                loss_vector_batch.append([self.pipeline_per_sample(sample, output) for sample, output in zip(list_batches_X[batch_number], list_batches_y[batch_number])])
                # Perform optimization algorithm (gradient descent)
                pass
            loss_vector_history.append(loss_vector_batch)
        # Save schema of network to JSON
        with open("schema.json", "w") as f:
            json.dump(self.get_schema(), f, ensure_ascii=False, indent=4)
        return loss_vector_history
        
    def pprint(self):
        print(f"Number of layers: {self.n_hidden + 2} Number of outputs: {self.n_outputs}")
        print("\n============================================\n")
        for layer_number, layer in enumerate(self.__network):
            print(f"Layer number: {layer_number + 1} Number of nodes: {layer.nodes}")
            for neuron_number, neuron in enumerate(layer.layer):
                print(f"Neuron number:{neuron_number + 1} Activation: {neuron.A} Weights: {neuron.W} Bias: {neuron.b}")
                # print("\n")
            print("---------------------------------------------------\n")

    def get_schema(self):
        dict_schema = {"n_layers": self.n_hidden + 2}
        for layer_number, layer in enumerate(self.__network):
            dict_layer = {"n_neurons": layer.nodes, "weights": [], "biases": [], "activation": []}
            for neuron in layer.layer:
                dict_layer["weights"].append(json.dumps(neuron.W, cls=NumpyArrayEncoder))
                dict_layer["biases"].append(json.dumps(neuron.b, cls=NumpyArrayEncoder))
                dict_layer["activation"].append(json.dumps(neuron.A, cls=NumpyArrayEncoder))
            dict_schema[layer_number + 1] = dict_layer
        return(dict_schema)


# class DisplayNetwork(Resource):
