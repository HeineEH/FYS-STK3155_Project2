from __future__ import annotations

# Typing
from .typing_utils import ArrayF, NetworkParams
from typing import TYPE_CHECKING, TypeVar
if TYPE_CHECKING:
    import numpy as np  # typed NumPy for the checker
else:
    import autograd.numpy as np  # runtime


T = TypeVar("T", float, ArrayF)
def runge(x: T) -> T:
    return 1/(1 + 25*x**2)

def generate_dataset(num = 500, noise = 0.05):
    x = np.random.uniform(-1, 1, num).reshape(-1, 1)
    y = runge(x) + noise*np.random.normal(0, 1, (num, 1))
    return x, y

def check_layer_params_equality(params1: NetworkParams, params2: NetworkParams) -> bool:
    if len(params1) != len(params2):
        raise ValueError("params1 and params2 must have the same number of layers")
    
    for (p1_W, p1_b), (p2_W, p2_b) in zip(params1, params2):
        if p1_W.shape != p2_W.shape or p1_b.shape != p2_b.shape:
            raise ValueError("Layer parameter shapes do not match")

        if not (np.allclose(p1_W, p2_W) and np.allclose(p1_b, p2_b)):
            return False
        
    return True