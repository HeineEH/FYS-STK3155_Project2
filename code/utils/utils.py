from __future__ import annotations

# Typing
from .typing_utils import ArrayF
from typing import TYPE_CHECKING, TypeVar
if TYPE_CHECKING:
    import numpy as np  # typed NumPy for the checker
else:
    import autograd.numpy as np  # runtime


T = TypeVar("T", float, ArrayF)
def runge(x: T) -> T:
    return 1/(1 + 25*x**2)

def generate_dataset(num = 400, noise = 0.05):
    x = np.linspace(-3, 3, num).reshape(-1, 1)
    np.random.shuffle(x)
    y = runge(x) + noise*np.random.normal(0, 1, (num, 1))
    return x, y