from __future__ import annotations
from autograd import grad # pyright: ignore[reportUnknownVariableType]

# Typing
from .typing_utils import ArrayF, NetworkParams
from .activation_functions import ActivationFunction
from .cost_functions import MSE, CostFunction
from utils.training import TrainingMethod
from typing import TYPE_CHECKING, Sequence, cast
if TYPE_CHECKING:
    import numpy as np  # typed NumPy for the checker
else:
    import autograd.numpy as np  # runtime


mse = MSE()
class NeuralNetwork:
    def __init__(
        self,
        network_input_size: int,
        layer_output_sizes: list[int],
        activation_funcs: Sequence[ActivationFunction],
        cost_fun: CostFunction,
        layers_random_state: int | None = None
    ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.cost_fun = cost_fun
        self.layers = self.create_layers_batch(random_state=layers_random_state)
        
    def create_layers_batch(self, random_state: int | None = None):
        rng = np.random.default_rng(random_state)

        layers: NetworkParams = []
        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = rng.standard_normal((layer_output_size, i_size)).T
            b = np.ones(layer_output_size)*0.01
            layers.append((W, b))
            i_size = layer_output_size
        return layers

    def reset_layers(self, random_state: int | None = None):
        self.layers = self.create_layers_batch(random_state=random_state)

    def predict(self, inputs: ArrayF):
        # Simple feed forward pass
        return self.feed_forward_batch(inputs)

    def mse_batch(self, inputs: ArrayF, targets: ArrayF) -> np.floating:
        predict = self.feed_forward_batch(inputs)
        return mse(predict, targets)

    def cost_batch(self, inputs: ArrayF, targets: ArrayF, include_regularization: bool = False) -> np.floating:
        predict = self.feed_forward_batch(inputs)
        cost = self.cost_fun(predict, targets)
        if include_regularization:
            flattened_params = self.flatten_params(self.layers)
            cost += self.cost_fun.apply_regularization(flattened_params)
        return cost

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
    
    def backpropagation_batch(self, input: ArrayF, target: ArrayF, layers: NetworkParams):
        layer_inputs, zs, predict = self.feed_forward_saver_batch(input)

        layer_grads: NetworkParams = []

        # We loop over the layers, from the last to the first
        dC_dz = None # Initialize this here to avoid a "possibly unbound" type-error.
        for i in reversed(range(len(layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_funcs[i].derivative

            if i == len(layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_fun.derivative(predict, target)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W_prev, _b_prev) = layers[i + 1]
                assert dC_dz is not None
                dC_da = dC_dz @ W_prev.T

            W, b = layers[i]
            dC_dz = dC_da*activation_der(z)
            dC_dW = layer_input.T @ dC_dz + self.cost_fun.apply_regularization_derivative(W)  # Shape (in_features, out_features)
            dC_db = np.sum(dC_dz, axis=0) + self.cost_fun.apply_regularization_derivative(b) # Sum over the batch axis to get shape (out_features,)

            layer_grads.append((dC_dW, dC_db))

        layer_grads.reverse()
        return layer_grads

    def compute_gradient(self, inputs: ArrayF, targets: ArrayF, layers: NetworkParams):
        return self.backpropagation_batch(inputs, targets, layers)

    def train(self, GD_method: TrainingMethod, num_iterations: int, n_batches: int = 5, track_mse = False):
        return GD_method.train(self.compute_gradient, self.layers, iterations=num_iterations, n_batches=n_batches, mse_track_func=(self.mse_batch if track_mse else None))
    
    # These last two methods are not needed in the project, but they can be nice to have! The first one has a layers parameter so that you can use autograd on it
    def autograd_compliant_predict(self, layers: NetworkParams, inputs: ArrayF, activation_funcs: Sequence[ActivationFunction]):
        a = inputs
        for (W, b), activation_func in zip(layers, activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a
    
    def autograd_compliant_cost(self, layers: NetworkParams, inputs: ArrayF, activation_funcs: Sequence[ActivationFunction], targets: ArrayF):
        prediction = self.autograd_compliant_predict(layers, inputs, activation_funcs)
        flattened_params = self.flatten_params(layers)
        cost = self.cost_fun(prediction, targets) + self.cost_fun.apply_regularization(flattened_params)
        return cost

    def autograd_gradient(self, inputs: ArrayF, targets: ArrayF) -> NetworkParams:
        autograd_func = grad(self.autograd_compliant_cost, 0) # type: ignore
        autograd_layer_grads = autograd_func(self.layers, inputs, self.activation_funcs, targets) # type: ignore
        return cast(NetworkParams, autograd_layer_grads) # cast to NetworkParams to get correct return type
    
    def flatten_params(self, layers: NetworkParams) -> ArrayF:
        parts: list[ArrayF] = []
        for W, b in layers:
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)