from typing import Literal
from abc import ABC, abstractmethod
import numpy as np

class _CostFunction(ABC):
    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass

class MSE(_CostFunction):
    def __init__(self, regularization: None | Literal["L1", "L2"] = None, lambd: None | float = None):
        use_regularization = regularization in ("L1", "L2")
        
        if not (regularization is None or use_regularization):
            raise ValueError("regularization must be None, 'L1', or 'L2'")

        if regularization in ("L1", "L2") and lambd is None:
            raise ValueError("the regularization parameter lambd must be provided when regularization is used")

        
        self.regularization = regularization
        self.lambd = lambd
    
    def __call__(self, y_true, y_pred, weights=None):
        mse = ((y_true - y_pred) ** 2).mean()
        
        # TODO: Add regularization

        
        return mse