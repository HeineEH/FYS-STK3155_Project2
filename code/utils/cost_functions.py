from __future__ import annotations
from typing import Literal
from abc import ABC, abstractmethod
from .activation_functions import Softmax

# Typing
from .typing_utils import ArrayF
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np  # typed NumPy for the checker
else:
    import autograd.numpy as np  # runtime

class CostFunction(ABC):
    regularization: None | Literal["L1", "L2"]

    def __init__(self, regularization: None | Literal["L1", "L2"] = None, lambd: float = 0):
        if not (regularization is None or regularization in ("L1", "L2")):
            raise ValueError("regularization must be None, 'L1', or 'L2'")

        if regularization and lambd == 0:
            raise ValueError("the regularization parameter lambd must be provided when regularization is used")

        
        self.regularization = regularization
        self.lambd = lambd

    @abstractmethod
    def __call__(self, y_pred: ArrayF, y_true: ArrayF) -> np.floating: ...

    @abstractmethod
    def derivative(self, y_pred: ArrayF, y_true: ArrayF) -> ArrayF:
        """Derivative with respect to `y_pred`"""
        ...

    def _l1(self, params: ArrayF) -> np.floating: return self.lambd*np.sum(np.abs(params))
    def _l1_derivative(self, params: ArrayF) -> ArrayF: return self.lambd*np.sign(params)

    def _l2(self, params: ArrayF) -> np.floating: return self.lambd*np.sum(params**2)
    def _l2_derivative(self, params: ArrayF) -> ArrayF: return 2*self.lambd*params

    def apply_regularization(self, params: ArrayF):
        # L1 regularization
        if self.regularization == "L1":
            return self._l1(params)
        
        # L2 regularization
        elif self.regularization == "L2":
            return self._l2(params)
        
        return 0. # No regularization applied
    
    def apply_regularization_derivative(self, params: ArrayF) -> float | ArrayF:
        # L1 regularization
        if self.regularization == "L1":
            return self._l1_derivative(params)
        
        # L2 regularization
        elif self.regularization == "L2":
            return self._l2_derivative(params)
        
        return 0.
        

class MSE(CostFunction):
    def __call__(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)
    
    def derivative(self, y_pred, y_true):
        return (2 / y_true.size) * (y_pred-y_true)


class BinaryCrossEntropy(CostFunction):
    def __call__(self, y_pred, y_true):
        return -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)) / y_true.size

    def derivative(self, y_pred, y_true):
        return ((1-y_true)/(1-y_pred) - y_true/y_pred) / y_true.size


class MulticlassCrossEntropy(CostFunction):
    def __call__(self, y_pred, y_true) -> np.floating:
        cross_entropy = -np.sum(y_true*np.log(y_pred)) / y_true.shape[0]
        return cross_entropy

    def derivative(self, y_pred, y_true):
        raise NotImplementedError("Derivative of MulticlassCrossEntropy is not implemented.")

softmax = Softmax()
def softmax_crossentropy_derivative(logits: ArrayF, y_true: ArrayF) -> ArrayF:
    probs = softmax(logits)
    return (probs - y_true) / y_true.shape[0]