import numpy
from numpy.typing import NDArray
import autograd.numpy as np # type: ignore
np: numpy = np # type: ignore . Workaround to not get type errors when using autograd's numpy wrapper.

from numpy.typing import NDArray
from typing import TypeVar

T = TypeVar("T", float, NDArray[numpy.floating])
def runge(x: T) -> T:
    return 1/(1 + 25*x**2)

def generate_dataset(num = 400, noise = 0.05):
    x = np.linspace(-3, 3, num).reshape(-1, 1)
    np.random.shuffle(x)
    y = runge(x) + noise*np.random.normal(0, 1, (num, 1))
    return x, y