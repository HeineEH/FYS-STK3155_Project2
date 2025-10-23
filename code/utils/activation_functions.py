from abc import ABC, abstractmethod
import numpy
from numpy.typing import NDArray
import autograd.numpy as np # type: ignore
np: numpy = np # type: ignore . Workaround to not get type errors when using autograd's numpy wrapper.


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, z: NDArray[numpy.float64]) -> NDArray[numpy.float64]: ...

    @abstractmethod
    def derivative(self, z: NDArray[numpy.float64]) -> NDArray[numpy.float64]: ...


class Sigmoid(ActivationFunction):
    def __call__(self, z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z):
        sig = self.__call__(z)
        return sig * (1 - sig)
    
class Identity(ActivationFunction):
    def __call__(self, z):
        return z
    
    def derivative(self, z):
        return np.ones_like(z)

class ReLU(ActivationFunction):
    def __call__(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return (np.where(z > 0, 1, 0))
    

class Softmax(ActivationFunction):
    def __call__(self, z):
        z_shift = z - np.max(z, axis=1, keepdims=True)  # To avoid taking exp of too large numbers. Gives same result.
        exp_z = np.exp(z_shift) 
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def derivative(self, z): # type: ignore
        raise NotImplementedError("Derivative of Softmax is not implemented. Use `SoftmaxCrossEntropy` cost function instead, with no last layer activation (identity func).")