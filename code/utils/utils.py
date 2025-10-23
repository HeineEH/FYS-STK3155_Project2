import numpy as np
from numpy.typing import NDArray
from typing import TypeVar

T = TypeVar("T", float, NDArray[np.float64])
def runge(x: T) -> T:
    return 1/(1 + 25*x**2)

def generate_dataset(num: int = 400):
    x = np.linspace(-1, 1, num)
    np.random.shuffle(x)
    y = runge(x) + 0.05*np.random.normal(0, 1, num)
    return x, y