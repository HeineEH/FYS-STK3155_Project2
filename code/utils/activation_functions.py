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

        Parameters
        ----------
        z : ArrayF
            Pre-activation input array (batch x features).

        Returns
        -------
        ArrayF
            Activated output (same shape as z).
        """
        ...

    @abstractmethod
    def derivative(self, z: ArrayF) -> ArrayF:
        """
        Evaluate the derivative of the activation function.

        Parameters
        ----------
        z : ArrayF
            Pre-activation input array (batch x features).

        Returns
        -------
        ArrayF
            Elementwise derivative (same shape as z).
        """
        ...


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.

    Used both in hidden layers, and output layer for binary classification.
    """
    def __call__(self, z: ArrayF):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z: ArrayF):
        sig = self.__call__(z)
        return sig * (1 - sig)
    
class Identity(ActivationFunction):
    """Identity activation function."""
    def __call__(self, z: ArrayF):
        return z
    
    def derivative(self, z: ArrayF):
        return np.ones_like(z)

class ReLU(ActivationFunction):
    """Rectified Linear Unit (ReLU) activation function."""
    def __call__(self, z: ArrayF):
        return np.maximum(0, z)

    def derivative(self, z: ArrayF):
        return (np.where(z > 0, 1, 0))
    
class LeakyReLU(ActivationFunction):
    """Leaky ReLU: keeps a small slope for negative inputs."""
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
    *Note:* The derivative is not implemented here due to its complexity. However, backpropagation will still work when used with the `MulticlassCrossEntropy` cost function.
    """
    def __call__(self, z: ArrayF):
        z_shift = z - np.max(z, axis=1, keepdims=True)  # To avoid taking exp of too large numbers. Gives same result.
        exp_z = np.exp(z_shift)
        s: ArrayF = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return s

    def derivative(self, z: ArrayF): # type: ignore
        raise NotImplementedError("Derivative of Softmax is not implemented.")