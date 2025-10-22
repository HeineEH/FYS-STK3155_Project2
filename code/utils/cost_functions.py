from typing import Literal
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from activation_functions import Softmax
import numpy
import autograd.numpy as np # type: ignore
np: numpy = np # type: ignore . Workaround to not get type errors when using autograd's numpy wrapper.

class _CostFunction(ABC):
    regularization: None | Literal["L1", "L2"]

    def __init__(self, regularization: None | Literal["L1", "L2"] = None, lambd: None | float = None):
        if not (regularization is None or regularization in ("L1", "L2")):
            raise ValueError("regularization must be None, 'L1', or 'L2'")

        if regularization and lambd is None:
            raise ValueError("the regularization parameter lambd must be provided when regularization is used")

        
        self.regularization = regularization
        self.lambd = lambd

    @abstractmethod
    def __call__(self, y_pred, y_true, params: None | NDArray = None) -> float:
        pass

    @abstractmethod
    def derivative(self, y_pred, y_true):
        """Derivative with respect to `y_pred`"""
        pass

    def _l1(self, params: NDArray): return self.lambd*np.sum(np.abs(params))
    def _l2(self, params: NDArray): return self.lambd*np.sum(params**2)

    def apply_regularization(self, params: None | NDArray):
        if self.regularization and params is None:
            raise ValueError(f"params must be provided when using regularization ({self.regularization})")

        # L1 regularization
        if self.regularization == "L1" and params is not None:
            return self._l1(params)
        
        # L2 regularization
        elif self.regularization == "L2" and params is not None:
            return self._l2(params)
        
        return 0 # No regularization applied


class MSE(_CostFunction):
    def __call__(self, y_pred, y_true, params: None | NDArray = None):
        if self.regularization and params is None:
            raise ValueError("params must be provided when using regularization")
        
        return np.mean((y_true - y_pred) ** 2) + self.apply_regularization(params)
    
    def derivative(self, y_pred, y_true):
        return (2 / y_true.size) * (y_pred-y_true)


class BinaryCrossEntropy(_CostFunction):
    def __call__(self, y_pred, y_true, params: None | NDArray = None):
        if self.regularization and params is None:
            raise ValueError("params must be provided when using regularization")
        
        cross_entropy = -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)) / y_true.size

        return cross_entropy + self.apply_regularization(params)

    def derivative(self, y_pred, y_true):
        return ((1-y_true)/(1-y_pred) - y_true/y_pred) / y_true.size


class SoftmaxCrossEntropy(_CostFunction):
    """Multiclass cross-entropy with softmax activation included"""

    softmax = Softmax()
    
    def __call__(self, y_pred, y_true, params: None | NDArray = None):
        if self.regularization and params is None:
            raise ValueError("params must be provided when using regularization")
        
        # y_pred is now the pre-activation z-values for the last layer
        # First apply softmax activation.
        probs = self.softmax(y_pred)

        # Then compute cross-entropy
        eps = 1e-12  # prevent log(0)
        cross_entropy = - np.sum(y_true*np.log(probs+eps)) / y_true.shape[0]

        return cross_entropy + self.apply_regularization(params)

    def derivative(self, y_pred, y_true):
        probs = self.softmax(y_pred)
        return (probs-y_true) / y_true.shape[0]