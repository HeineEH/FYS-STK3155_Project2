from __future__ import annotations
from autograd import grad # pyright: ignore[reportUnknownVariableType]
from .cost_functions import MSE, MulticlassCrossEntropy, softmax_crossentropy_derivative

# Typing
from .typing_utils import ArrayF, NetworkParams
from .activation_functions import ActivationFunction, Softmax
from .cost_functions import CostFunction
from utils.training import TrainingMethod
from typing import TYPE_CHECKING, Sequence, cast
if TYPE_CHECKING:
    import numpy as np  # typed NumPy for the checker
else:
    import autograd.numpy as np  # runtime


mse = MSE()
class NeuralNetwork:
    """
    Feedforward neural network with backpropagation and training.

    Parameters
    ----------
    network_input_size : int
        Number of input features.
    layer_output_sizes : list[int]
        Output dimension of each hidden / output layer (in order).
    activation_funcs : Sequence[ActivationFunction]
        Activation functions for each layer (same length as layer_output_sizes).
    cost_fun : CostFunction
        Cost function instance to use for training.
    layers_random_state : int | None, optional
        Seed used for random initialization of layer parameters.
    """

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


        # Check if Softmax and MulticlassCrossEntropy are used
        uses_softmax = isinstance(activation_funcs[-1], Softmax)
        uses_cross_entropy = isinstance(cost_fun, MulticlassCrossEntropy)
        if uses_softmax != uses_cross_entropy:
            raise ValueError("Softmax activation function must be used with MulticlassCrossEntropy cost function, and vice versa.")
        self.softmax_crossentropy_special_case = uses_softmax and uses_cross_entropy

        
    def create_layers_batch(self, random_state: int | None = None):
        """Get initial layer parameters."""
        rng = np.random.default_rng(random_state)

        layers: NetworkParams = []
        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = rng.standard_normal((layer_output_size, i_size)).T # We really want (i_size, layer_output_size) so transpose
            b = np.ones(layer_output_size)*0.01 # Initialize biases to small constant to avoid dead neurons when using ReLU
            layers.append((W, b))
            i_size = layer_output_size # The next layer's input size is this layer's output size
        return layers

    def reset_layers(self, random_state: int | None = None):
        """
        Reset parameters by re-initializing layers.

        Parameters
        ----------
        random_state : int | None, optional
            RNG seed.
        """
        self.layers = self.create_layers_batch(random_state=random_state)

    def mse_batch(self, inputs: ArrayF, targets: ArrayF) -> np.floating:
        """Compute MSE over a batch."""
        predict = self.feed_forward_batch(inputs)
        return mse(predict, targets)
    
    def accuracy_batch(self, inputs: ArrayF, targets: ArrayF) -> np.floating:
        """Compute classification accuracy for one-hot encoded targets."""
        predict = self.feed_forward_batch(inputs)

        # The predictions and targets are one-hot encoded, so we take the argmax to get class labels
        predicted_classes = np.argmax(predict, axis=1) 
        true_classes = np.argmax(targets, axis=1)

        # predicted_classes == true_classes gives a boolean array, it evaluates to 1 for True 
        # and 0 for False, so we can take the mean to get accuracy.
        accuracy = np.mean((predicted_classes == true_classes).astype(int), dtype=np.float64)
        return accuracy

    def cost_batch(self, inputs: ArrayF, targets: ArrayF, include_regularization: bool = False) -> np.floating:
        """Compute the cost for a batch, optionally including regularization."""
        predict = self.feed_forward_batch(inputs)
        cost = self.cost_fun(predict, targets)
        if include_regularization:
            flattened_params = self.flatten_params(self.layers)
            cost += self.cost_fun.apply_regularization(flattened_params)
        return cost

    def feed_forward_batch(self, inputs: ArrayF):
        """
        Standard batch feed-forward pass.

        Parameters
        ----------
        inputs : ArrayF
            Input array (batch, input_dim).

        Returns
        -------
        ArrayF
            Output activations from the final layer (batch, output_dim).
        """
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a
    predict = feed_forward_batch # Alias for predict method

    def feed_forward_saver_batch(self, input: ArrayF):
        """
        Feed-forward that also returns inputs and pre-activations for each layer.

        Used for manual backpropagation.

        Parameters
        ----------
        input : ArrayF
            Input array (batch, input_dim).

        Returns
        -------
        tuple[list[ArrayF], list[ArrayF], ArrayF]
            (layer_inputs, zs, final_output) where:
            - layer_inputs: list of inputs for each layer.
            - zs: list of pre-activation arrays for each layer.
            - final_output: activations of the last layer.
        """
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
        """
        Compute gradients for a batch using backpropagation.

        Parameters
        ----------
        input : ArrayF
            Input array (batch, input_dim).
        target : ArrayF
            Target array (batch, output_dim).
        layers : NetworkParams
            List of (W, b) parameter tuples to compute gradients of.
        """

        layer_inputs, zs, predict = self.feed_forward_saver_batch(input)

        layer_grads: NetworkParams = []

        # We loop over the layers, from the last to the first
        dC_dz = None # Initialize this here to avoid a "possibly unbound" type-error.
        for i in reversed(range(len(layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_funcs[i].derivative

            if i == len(layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly

                if self.softmax_crossentropy_special_case:
                    # Special case for MulticlassCrossEntropy with Softmax activation
                    dC_dz = softmax_crossentropy_derivative(z, target)
                else:
                    dC_da = self.cost_fun.derivative(predict, target)
                    dC_dz = dC_da*activation_der(z)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W_prev, _b_prev) = layers[i + 1]
                assert dC_dz is not None
                dC_da = dC_dz @ W_prev.T
                dC_dz = dC_da*activation_der(z)

            W, b = layers[i]

            # Add regularization derivative to gradients (only adds if regularization is defined on the cost function object)
            dC_dW = layer_input.T @ dC_dz + self.cost_fun.apply_regularization_derivative(W)  # Shape (in_features, out_features)
            dC_db = np.sum(dC_dz, axis=0) + self.cost_fun.apply_regularization_derivative(b) # Sum over the batch axis to get shape (out_features,)

            layer_grads.append((dC_dW, dC_db))

        # Reverse to get correct order
        layer_grads.reverse()

        return layer_grads
    compute_gradient = backpropagation_batch # Alias for compute_gradient method

    def train(
            self,
            GD_method: TrainingMethod,
            num_iterations: int,
            n_batches: int = 5,
            track_mse: bool = False,
            track_accuracy: bool = False,
            verbose: bool = False
    ) -> ArrayF:
        """
        Train the network using the provided TrainingMethod

        Parameters
        ----------
        GD_method : TrainingMethod
            Traning method object implementing a `train` method.
        num_iterations : int
            Number of training iterations (epochs).
        n_batches : int, default=5
            Number of mini-batches to split the data into (passed to GD_method).
        track_mse : bool, optional
            If True track and return MSE history.
        track_accuracy : bool, optional
            If True track and return accuracy history.
        verbose : bool, optional
            If True print progress information.
        
        Returns
        -------
        ArrayF
            Return value from GD_method.train (tracking history if enabled).
        """
        
        if track_mse and track_accuracy:
            raise ValueError("Cannot track both MSE and accuracy at the same time.")
        
        if track_mse:
            track_func = self.mse_batch
        elif track_accuracy:
            track_func = self.accuracy_batch
        else:
            track_func = None

        return GD_method.train(self.compute_gradient, self.layers, iterations=num_iterations, n_batches=n_batches, track_func=track_func, verbose=verbose)
    
    # These last two methods are not needed in the project, but they can be nice to have! The first one has a layers parameter so that you can use autograd on it
    def autograd_compliant_predict(self, layers: NetworkParams, inputs: ArrayF, activation_funcs: Sequence[ActivationFunction]):
        """Predict using an explicit `layers` parameter for autograd compatibility."""
        a = inputs
        for (W, b), activation_func in zip(layers, activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a
    
    def autograd_compliant_cost(self, layers: NetworkParams, inputs: ArrayF, activation_funcs: Sequence[ActivationFunction], targets: ArrayF):
        """Cost function compatible with autograd. Includes regularization."""
        prediction = self.autograd_compliant_predict(layers, inputs, activation_funcs)
        flattened_params = self.flatten_params(layers)
        cost = self.cost_fun(prediction, targets) + self.cost_fun.apply_regularization(flattened_params)
        return cost

    def autograd_gradient(self, inputs: ArrayF, targets: ArrayF) -> NetworkParams:
        """
        Compute gradients using autograd by differentiating autograd_compliant_cost.

        Parameters
        ----------
        inputs : ArrayF
            Input array (batch, input_dim).
        targets : ArrayF
            Target array (batch, output_dim).

        Returns
        -------
        NetworkParams
            Gradients in the same structure as `self.layers`.
        """  
        autograd_func = grad(self.autograd_compliant_cost, 0) # type: ignore
        autograd_layer_grads = autograd_func(self.layers, inputs, self.activation_funcs, targets) # type: ignore
        return cast(NetworkParams, autograd_layer_grads) # cast to NetworkParams to get correct return type
    
    def flatten_params(self, layers: NetworkParams) -> ArrayF:
        """Flatten parameters to a 1D array."""
        parts: list[ArrayF] = []
        for W, b in layers:
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)
    
    def __str__(self):
        """String representation of the neural network with layer summary."""
        s = "Neural Network:"
        s += f"\n\tLayers:\t\t {self.network_input_size}"
        for layer in zip(self.layer_output_sizes, self.activation_funcs):
            s += f" -> {layer[0]} ({type(layer[1]).__name__})"

        s += f"\n\tCost function:\t {type(self.cost_fun).__name__}"
        if self.cost_fun.regularization is not None:
            s += f"\n\tRegularization:\t {self.cost_fun.regularization} (λ={self.cost_fun.lambd:.1e})\n"

        return s