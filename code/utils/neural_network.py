import numpy
import autograd.numpy as np # type: ignore
np: numpy = np # type: ignore . Workaround to not get type errors when using autograd's numpy wrapper.

from utils.training import TrainingMethod
from typing import Sequence, cast
from numpy.typing import NDArray
from autograd import grad # pyright: ignore[reportUnknownVariableType]
from .activation_functions import ActivationFunction
from .cost_functions import CostFunction

# Type aliases
ArrayF = NDArray[numpy.floating]
Layer = tuple[ArrayF, ArrayF]
Layers = list[Layer]


class NeuralNetwork:
    def __init__(
        self,
        network_input_size: int,
        layer_output_sizes: list[int],
        activation_funcs: Sequence[ActivationFunction],
        cost_fun: CostFunction,
    ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.cost_fun = cost_fun
        self.layers = self.create_layers_batch()
        
    def create_layers_batch(self):
        layers: Layers = []
        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.randn(layer_output_size, i_size).T
            b = np.ones(layer_output_size)*0.01
            layers.append((W, b))
            i_size = layer_output_size
        return layers

    def predict(self, inputs: ArrayF):
        # Simple feed forward pass
        return self.feed_forward_batch(inputs)

    def cost_batch(self, inputs: ArrayF, targets: ArrayF):
        predict = self.feed_forward_batch(inputs)
        return self.cost_fun(predict, targets)

    def feed_forward_batch(self, inputs: ArrayF):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def feed_forward_saver_batch(self, input: ArrayF):
        layer_inputs: list[ArrayF] = []
        zs: list[ArrayF] = []
        a = input
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)

            zs.append(z)

        return layer_inputs, zs, a
    
    def backpropagation_batch(self, input: ArrayF, target: ArrayF, layers: Layers):
        layer_inputs, zs, predict = self.feed_forward_saver_batch(input)

        layer_grads: Layers = []

        # We loop over the layers, from the last to the first
        dC_dz = None # Initialize this here to avoid a "possibly unbound" type-error.
        for i in reversed(range(len(layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_funcs[i].derivative

            if i == len(layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_fun.derivative(predict, target)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, _b) = layers[i + 1]
                assert dC_dz is not None
                dC_da = dC_dz @ W.T

            dC_dz = dC_da*activation_der(z)
            dC_dW = layer_input.T @ dC_dz
            dC_db = np.sum(dC_dz, axis=0) # Sum over the batch axis to get shape (out_features,)

            layer_grads.append((dC_dW, dC_db))

        layer_grads.reverse()
        return layer_grads

    def compute_gradient(self, inputs: ArrayF, targets: ArrayF, layers: Layers):
        return self.backpropagation_batch(inputs, targets, layers)

    def train(self, GD_method: TrainingMethod, num_iterations: int, n_batches: int = 5):
        self.layers = GD_method.train(self.compute_gradient, self.layers, num_iterations, n_batches)

    # These last two methods are not needed in the project, but they can be nice to have! The first one has a layers parameter so that you can use autograd on it
    def autograd_compliant_predict(self, layers: Layers, inputs: ArrayF, activation_funcs: Sequence[ActivationFunction]):
        a = inputs
        for (W, b), activation_func in zip(layers, activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a
    
    def autograd_compliant_cost(self, layers: Layers, inputs: ArrayF, activation_funcs: Sequence[ActivationFunction], targets: ArrayF):
        prediction = self.autograd_compliant_predict(layers, inputs, activation_funcs)
        cost = self.cost_fun(prediction, targets)
        return cost

    def autograd_gradient(self, inputs: ArrayF, targets: ArrayF):
        autograd_layer_grads = cast(Layers, grad(self.autograd_compliant_cost, 0)(self.layers, inputs, self.activation_funcs, targets))
        return autograd_layer_grads