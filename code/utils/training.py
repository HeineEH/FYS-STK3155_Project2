import numpy as nup
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from typing import TYPE_CHECKING
from sklearn.model_selection import train_test_split
from sklearn import linear_model

if TYPE_CHECKING:
    from .step_methods import _StepMethod

# Template for training methods, like gradient descent, and stochastic gradient descent
class _TrainingMethod:
    def __init__(
        self,
        step_method: "_StepMethod",
        inputs, 
        targets,
    ) -> None:
        self.step_method = step_method
        self.step_method.caller = self
        self.inputs = inputs
        self.targets = targets
        self.step_method.caller = self
        self.setup()

    def setup(self):
        ...
    
    def train(self, *args, **kwargs) -> tuple[npt.ArrayLike, npt.ArrayLike] | None:
        ...


# ========== Training methods ==========

class GradientDescent(_TrainingMethod):
    def train(self, gradient, layers: npt.ArrayLike, iterations: int = 1000,n_batches = 5) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        self.step_method.setup(layers)
        for i in range(iterations):
            layers_grad = gradient(self.inputs, self.targets,layers)
            layers = self.step_method.train_step(layers_grad,layers) 
        return layers
                
class StochasticGradientDescent(_TrainingMethod): 
    def learning_schedule(self,t,t0,t1): 
        return t0/(t + t1)

    def train(self, gradient, layers, epochs: int = 1000, n_batches: int = 5) -> tuple[npt.ArrayLike, npt.ArrayLike]:

        n_datapoints = self.inputs.shape[0]
        batch_size = int(n_datapoints/n_batches)
        initial_learning_rate = self.step_method.learning_rate/n_batches   # divide by number of batches to take care of bias in cost functions
        self.step_method.setup(layers)
        
        for i in range(epochs):
            shuffled_data = nup.array(range(n_datapoints))
            nup.random.shuffle(shuffled_data)
            for j in range(n_batches): 
                layers_grad = gradient(self.inputs[shuffled_data][(batch_size*j):(batch_size*(j+1))], self.targets[shuffled_data][(batch_size*j):(batch_size*(j+1))],layers)
                t = i*n_batches + j
                self.step_method.learning_rate = self.learning_schedule(t,initial_learning_rate*70*n_batches,70*n_batches)
                layers = self.step_method.train_step(layers_grad,layers) 
        return layers