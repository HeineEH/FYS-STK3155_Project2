import numpy
import autograd.numpy as np # type: ignore
np: numpy = np # type: ignore . Workaround to not get type errors when using autograd's numpy wrapper.

from autograd import grad, elementwise_grad
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from .activation_functions import _ActivationFunction
from .cost_functions import _CostFunction
from .training import _TrainingMethod
from .step_methods import _StepMethod

def normalize_input_target(inputs,targets,test_size = 0.0): 
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=test_size,random_state=35)
    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_inputs)
    test_inputs = scaler.transform(test_inputs)
    if test_size != 0: 
        return train_inputs, test_inputs, train_targets, test_targets
    else: 
        return train_inputs, train_targets

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs: list[_ActivationFunction],
        cost_fun: _CostFunction,
    ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.cost_fun = cost_fun
        self.layers = self.create_layers_batch()
        
    def create_layers_batch(self):
        layers = []
        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.randn(layer_output_size, i_size).T
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size
        return layers

    def predict(self, inputs):
        # Simple feed forward pass
        return self.feed_forward_batch(inputs)

    def cost_batch(self,inputs, targets):
        predict = self.feed_forward_batch(inputs)
        return self.cost_fun(predict, targets)

    def feed_forward_batch(self,inputs):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def feed_forward_saver_batch(self,input):
        layer_inputs = []
        zs = []
        a = input
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)

            zs.append(z)

        return layer_inputs, zs, a
    
    def backpropagation_batch(self,
        input,target, layers
    ):
        layer_inputs, zs, predict = self.feed_forward_saver_batch(input)

        layer_grads = [() for layer in layers]

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_funcs[i].derivative

            if i == len(layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_fun.derivative(predict, target)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = layers[i + 1]
                dC_da = dC_dz @ W.T

            dC_dz = dC_da*activation_der(z)
            dC_dW = layer_input.T @ dC_dz
            dC_db = np.sum(dC_dz, axis=0) # Sum over the batch axis to get shape (out_features,)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def compute_gradient(self, inputs, targets, layers):
        return self.backpropagation_batch(inputs, targets, layers)

    def train(self, GD_method,num_iterations,n_batches = 5):
        self.layers = GD_method.train(self.compute_gradient,self.layers,num_iterations,n_batches)

    # These last two methods are not needed in the project, but they can be nice to have! The first one has a layers parameter so that you can use autograd on it
    def autograd_compliant_predict(self, layers, inputs):
        pass

    def autograd_gradient(self, inputs, targets):
        layer_grads = self.compute_gradient(inputs,targets)
        cost_grad = grad(self.cost_batch, 0)
        autograd_layer_grads = cost_grad(self.layers, inputs, self.activation_funcs, targets)
        for i in range(len(self.layers)):
            equal = np.allclose(layer_grads[i][0], autograd_layer_grads[i][0]) and np.allclose(layer_grads[i][1], autograd_layer_grads[i][1])
            print(f"Layer {i}: Own implementation grads equal to autograd results: {equal}")