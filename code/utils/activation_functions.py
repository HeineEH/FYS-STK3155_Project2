from __future__ import annotations
from abc import ABC, abstractmethod

# Typing
from .typing_utils import ArrayF
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np  # typed NumPy for the checker
else:
    import autograd.numpy as np  # runtime


class ActivationFunction(ABC):
    """Abstract base class for activation functions."""
    @abstractmethod
    def __call__(self, z: ArrayF) -> ArrayF:
        """
        Evaluate the activation function.
        Parameters:
            z (NDArray[Float]): Input array.
        Returns:
            NDArray[Float]: Activated output array.
        """
        ...

    @abstractmethod
    def derivative(self, z: ArrayF) -> ArrayF:
        """
        Evaluate the derivative of the activation function with respect to the input.
        Parameters:
            z (NDArray[Float]): Input array.
        Returns:
            NDArray[Float]: Derivative of the activation function.
        """
        ...


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    f(z) = 1 / (1 + exp(-z))

    Used both in hidden layers, and output layer for binary classification.
    """
    def __call__(self, z: ArrayF):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z: ArrayF):
        sig = self.__call__(z)
        return sig * (1 - sig)
    
class Identity(ActivationFunction):
    """
    Identity activation function.
    f(z) = z
    """
    def __call__(self, z: ArrayF):
        return z
    
    def derivative(self, z: ArrayF):
        return np.ones_like(z)

class ReLU(ActivationFunction):
    """
    ReLU activation function.
    f(z) = max(0, z)
    """
    def __call__(self, z: ArrayF):
        return np.maximum(0, z)

    def derivative(self, z: ArrayF):
        return (np.where(z > 0, 1, 0))
    
class LeakyReLU(ActivationFunction):
    """
    Leaky ReLU activation function.
    f(z) = max(0, z) + negative_slope * min(0, z)
    """
    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope
    
    def __call__(self, z: ArrayF):
        return np.maximum(0, z) + self.negative_slope*np.minimum(0, z)

    def derivative(self, z: ArrayF):
        return (np.where(z > 0, 1, self.negative_slope))

class Softmax(ActivationFunction):
    """
    Softmax activation function.
    f(z_i) = exp(z_i) / sum_j[exp(z_j)]

    Typically used in the output layer for multi-class classification.
    
    Note: The derivative is not implemented here due to its complexity. However, backpropagation will still work when used with `MulticlassCrossEntropy`.
    """
    def __call__(self, z: ArrayF):
        z_shift = z - np.max(z, axis=1, keepdims=True)  # To avoid taking exp of too large numbers. Gives same result.
        exp_z = np.exp(z_shift)
        s: ArrayF = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return s

    def derivative(self, z: ArrayF): # type: ignore
        raise NotImplementedError("Derivative of Softmax is not implemented.")