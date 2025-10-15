import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
    ):
        pass
        
    def create_layers_batch(self,network_input_size, layer_output_sizes):
        layers = []

        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = np.random.randn(layer_output_size, i_size).T
            b = np.random.randn(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size
        return layers

    def predict(self, inputs):
        # Simple feed forward pass
        pass

    def cost(self, inputs, targets):
        pass

    def feed_forward_saver_batch(self,input, layers, activation_funcs):
        layer_inputs = []
        zs = []
        a = input
        for (W, b), activation_func in zip(layers, activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)

            zs.append(z)

        return layer_inputs, zs, a
    
    def backpropagation_batch(self,
        input, layers, activation_funcs, target, activation_ders, cost_der=...
    ):
        layer_inputs, zs, predict = self.feed_forward_saver_batch(input, layers, activation_funcs)

        layer_grads = [() for layer in layers]

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

            if i == len(layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = cost_der(predict, target)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = layers[i + 1]
                dC_da = dC_dz @ W.T

            dC_dz = dC_da*activation_der(z)
            dC_dW = layer_input.T @ dC_dz
            dC_db = np.sum(dC_dz, axis=0) # Sum over the batch axis to get shape (out_features,)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def compute_gradient(self, inputs, targets):
        pass

    def update_weights(self, layer_grads):
        pass

    # These last two methods are not needed in the project, but they can be nice to have! The first one has a layers parameter so that you can use autograd on it
    def autograd_compliant_predict(self, layers, inputs):
        pass

    def autograd_gradient(self, inputs, targets):
        pass