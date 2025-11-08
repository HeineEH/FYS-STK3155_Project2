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
    """Abstract base class for cost functions with optional regularization."""
    regularization: None | Literal["L1", "L2"]

    def __init__(self, regularization: None | Literal["L1", "L2"] = None, lambd: float = 0):
        if not (regularization is None or regularization in ("L1", "L2")):
            raise ValueError("regularization must be None, 'L1', or 'L2'")

        if regularization and lambd == 0:
            raise ValueError("the regularization parameter lambd must be provided when regularization is used")

        
        self.regularization = regularization
        self.lambd = lambd

    @abstractmethod
    def __call__(self, y_pred: ArrayF, y_true: ArrayF) -> np.floating:
        """
        Evaluate the cost for the provided predictions and targets, excluding regularization.

        Regularization contributions can be obtained via `apply_regularization` and added by the caller when needed.

        Parameters
        ----------
        y_pred : ArrayF
            Predicted output array (batch x output_dim).
        y_true : ArrayF
            True output array (batch x output_dim).

        Returns
        -------
        np.floating
            Scalar cost value averaged over the batch.
        """
        ...

    @abstractmethod
    def derivative(self, y_pred: ArrayF, y_true: ArrayF) -> ArrayF:
        """
        Compute the derivative of the cost with respect to the predictions `y_pred`.
        
        Parameters
        ----------
        y_pred : ArrayF
            Predicted output array (batch x output_dim).
        y_true : ArrayF
            True output array (batch x output_dim).

        Returns
        -------
        ArrayF
            Array of same shape as y_pred containing dC/d(y_pred).
        """
        ...

    # L1 regularization
    def _l1(self, params: ArrayF) -> np.floating: return self.lambd*np.sum(np.abs(params))
    def _l1_derivative(self, params: ArrayF) -> ArrayF: return self.lambd*np.sign(params)

    def _l2(self, params: ArrayF) -> np.floating: return self.lambd*np.sum(params**2)
    def _l2_derivative(self, params: ArrayF) -> ArrayF: return 2*self.lambd*params

    def apply_regularization(self, params: ArrayF):
        """
        Compute the regularization term for given flattened parameters.

        Parameters
        ----------
        params : ArrayF
            Flattened model parameters.

        Returns
        -------
        np.floating
            Regularization contribution to add to the cost (0. if none).
        """
        
        if self.regularization == "L1":
            return self._l1(params)
        
        elif self.regularization == "L2":
            return self._l2(params)
        
        return 0. # No regularization applied
    
    def apply_regularization_derivative(self, params: ArrayF) -> float | ArrayF:
        """
        Compute the derivative of the regularization term w.r.t. parameters.

        Parameters
        ----------
        params : ArrayF
            Flattened model parameters.

        Returns
        -------
        float | ArrayF
            Derivative of the regularization term (same shape as params), or 0. if none.
        """

        # L1 regularization
        if self.regularization == "L1":
            return self._l1_derivative(params)
        
        # L2 regularization
        elif self.regularization == "L2":
            return self._l2_derivative(params)
        
        return 0.
        

class MSE(CostFunction):
    def __call__(self, y_pred: ArrayF, y_true: ArrayF):
        return np.mean((y_true - y_pred) ** 2)
    
    def derivative(self, y_pred: ArrayF, y_true: ArrayF):
        return (2 / y_true.size) * (y_pred-y_true)


class BinaryCrossEntropy(CostFunction):
    def __call__(self, y_pred: ArrayF, y_true: ArrayF):
        return -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)) / y_true.size

    def derivative(self, y_pred: ArrayF, y_true: ArrayF):
        return ((1-y_true)/(1-y_pred) - y_true/y_pred) / y_true.size


class MulticlassCrossEntropy(CostFunction):
    def __call__(self, y_pred: ArrayF, y_true: ArrayF):
        cross_entropy = -np.sum(y_true*np.log(y_pred)) / y_true.shape[0]
        return cross_entropy

    def derivative(self, y_pred: ArrayF, y_true: ArrayF): # type: ignore
        raise NotImplementedError("Derivative of MulticlassCrossEntropy is not implemented.")

softmax = Softmax()
def softmax_crossentropy_derivative(logits: ArrayF, y_true: ArrayF) -> ArrayF:
    """
    Derivative of softmax combined with multiclass cross-entropy w.r.t. logits.

    This computes d/d(logits) using the shortcut: softmax(logits) - y_true, averaged over the batch.

    Parameters
    ----------
    logits : ArrayF
        Pre-softmax activations (batch x num_classes).
    y_true : ArrayF
        One-hot encoded true labels (batch x num_classes).

    Returns
    -------
    ArrayF
        Array of same shape as logits with the gradient of the loss w.r.t. logits.
    """
    probs = softmax(logits)
    return (probs - y_true) / y_true.shape[0]