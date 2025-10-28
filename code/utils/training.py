from __future__ import annotations
from abc import ABC, abstractmethod
import numba as nb

# Typing
from .typing_utils import ArrayF, NetworkParams
from .step_methods import StepMethod
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    import numpy as np  # typed NumPy for the checker
else:
    import autograd.numpy as np  # runtime


GradientFunc = Callable[[ArrayF, ArrayF, NetworkParams], NetworkParams]

# Template for training methods, like gradient descent, and stochastic gradient descent
class TrainingMethod(ABC):
    def __init__(
        self,
        step_method: "StepMethod",
        inputs: ArrayF,
        targets: ArrayF,
    ) -> None:
        self.step_method = step_method
        self.step_method.caller = self
        self.inputs = inputs
        self.targets = targets
        self.step_method.caller = self

    @abstractmethod
    def train(self, gradient: GradientFunc, layers: NetworkParams, iterations: int = 1000, n_batches: int = 5):
        """Train the neural network. Mutates the layers in place."""
        ...


# ========== Training methods ==========

class GradientDescent(TrainingMethod):
    
    @nb.njit(parallel=True)
    def train(self, gradient, layers, iterations = 1000, n_batches = 5):
        self.step_method.setup(layers)
        for _ in range(iterations):
            layers_grad = gradient(self.inputs, self.targets,layers)
            self.step_method.train_step(layers_grad,layers) 
                
class StochasticGradientDescent(TrainingMethod): 
    def learning_schedule(self, t: float, t0: float, t1: float): 
        return t0/(t + t1)

    @nb.njit(parallel=True)
    def train(self, gradient, layers, iterations = 1000, n_batches = 5):
        n_datapoints = self.inputs.shape[0]
        batch_size = int(n_datapoints/n_batches)
        initial_learning_rate = self.step_method.learning_rate/n_batches   # divide by number of batches to take care of bias in cost functions
        self.step_method.setup(layers)
        
        for i in range(iterations):
            shuffled_data = np.array(range(n_datapoints))
            np.random.shuffle(shuffled_data)
            for j in range(n_batches): 
                layers_grad = gradient(self.inputs[shuffled_data][(batch_size*j):(batch_size*(j+1))], self.targets[shuffled_data][(batch_size*j):(batch_size*(j+1))],layers)
                t = i*n_batches + j
                self.step_method.learning_rate = self.learning_schedule(t,initial_learning_rate*70*n_batches,70*n_batches)
                self.step_method.train_step(layers_grad,layers) 