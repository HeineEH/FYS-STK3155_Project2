from __future__ import annotations
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy
import autograd.numpy as np # type: ignore
np: numpy = np # type: ignore . Workaround to not get type errors when using autograd's numpy wrapper.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from training import TrainingMethod

# Template for step methods, like gd-momentum, RMSprop, ADAgrad
class StepMethod(ABC):
    caller: "TrainingMethod"
    learning_rate: float
    @abstractmethod
    def setup(self, starting_layers: NDArray[numpy.floating]) -> None: ...

    @abstractmethod
    def train_step(self, layers_grad: NDArray[numpy.floating], layers: NDArray[numpy.floating]) -> NDArray[numpy.floating]: ...



# ========== Step methods ==========

class ConstantLearningRateStep(StepMethod):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    def training_increment(self, gradient: NDArray[numpy.floating]):
        return self.learning_rate * gradient
    
    def train_step(self,layers_grad,layers):
        for (W,b), (W_g,b_g) in zip(layers, layers_grad): 
            W -= self.training_increment(W_g)
            b -= self.training_increment(b_g)
        return layers

class MomentumStep(StepMethod):
    def __init__(self, learning_rate: float, momentum: float) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def setup(self, starting_layers):
        self.velocity: list[tuple[NDArray[numpy.floating], NDArray[numpy.floating]]] = []
        for (W,b) in starting_layers:
            self.velocity.append((np.zeros_like(W), np.zeros_like(b)))
    
    def training_increment(self, gradient: NDArray[numpy.floating], velocity: NDArray[numpy.floating]):
        velocity = self.momentum * velocity + self.learning_rate * gradient
        return velocity
    
    def train_step(self, layers_grad, layers):
        for (W,b), (W_g,b_g), (W_vel,b_vel) in zip(layers, layers_grad, self.velocity): 
            W_vel = self.training_increment(W_g,W_vel)
            b_vel = self.training_increment(b_g,b_vel)
            W -= W_vel
            b -= b_vel
        return layers


class ADAgradStep(StepMethod):
    def __init__(self, learning_rate: float, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.error = error
    
    def setup(self, starting_layers):
        self.accumulated_gradient: list[tuple[NDArray[numpy.floating], NDArray[numpy.floating]]] = []
        for (W,b) in starting_layers:
            self.accumulated_gradient.append((np.zeros_like(W), np.zeros_like(b)))
        
    def training_increment(self, gradient: NDArray[numpy.floating], accumulated_gradient: NDArray[numpy.floating]):
        accumulated_gradient += gradient**2  # Accumulate squared gradients
        adjusted_gradient = gradient / (np.sqrt(accumulated_gradient) + self.error)
        return self.learning_rate * adjusted_gradient, accumulated_gradient
    
    def train_step(self, layers_grad, layers):
        for (W,b),(W_g,b_g),(W_g_acc,b_g_acc) in zip(layers, layers_grad, self.accumulated_gradient): 
            W_i, W_g_acc = self.training_increment(W_g, W_g_acc)
            b_i, b_g_acc = self.training_increment(b_g, b_g_acc)
            W -= W_i
            b -= b_i
        return layers

class RMSpropStep(StepMethod):
    def __init__(self, learning_rate: float, decay_rate: float, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.error = error
        
    def setup(self, starting_layers) -> None:
        self.accumulated_gradient: list[tuple[NDArray[numpy.floating], NDArray[numpy.floating]]] = []
        for (W,b) in starting_layers:
            self.accumulated_gradient.append((np.zeros_like(W), np.zeros_like(b)))
    
    def training_increment(self, gradient: NDArray[numpy.floating], accumulated_gradient: NDArray[numpy.floating]):
        accumulated_gradient = self.decay_rate * accumulated_gradient + (1 - self.decay_rate) * gradient**2
        adjusted_gradient = gradient / (np.sqrt(accumulated_gradient) + self.error)
        return self.learning_rate * adjusted_gradient, accumulated_gradient
    
    def train_step(self, layers_grad, layers):
        for (W,b),(W_g,b_g),(W_g_acc,b_g_acc) in zip(layers, layers_grad, self.accumulated_gradient): 
            W_i, W_g_acc = self.training_increment(W_g, W_g_acc)
            b_i, b_g_acc = self.training_increment(b_g, b_g_acc)
            W -= W_i
            b -= b_i
        return layers       
    
class AdamStep(StepMethod):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.error = error

    def setup(self, starting_layers) -> None:
        self.t = 0  # Time step
        self.s: list[tuple[NDArray[numpy.floating], NDArray[numpy.floating]]] = []   # First moment vector
        self.r: list[tuple[NDArray[numpy.floating], NDArray[numpy.floating]]] = []   # Second moment vector
        for (W,b) in starting_layers:
            self.s.append((np.zeros_like(W), np.zeros_like(b)))
            self.r.append((np.zeros_like(W), np.zeros_like(b)))
    
    def training_increment(self, gradient: NDArray[numpy.floating], r: NDArray[numpy.floating], s: NDArray[numpy.floating]):
        self.t += 1

        s = self.beta1 * s + (1 - self.beta1) * gradient
        r = self.beta2 * r + (1 - self.beta2) * (gradient ** 2)

        s_hat = s / (1 - self.beta1 ** self.t)
        r_hat = r / (1 - self.beta2 ** self.t)

        adjusted_gradient = s_hat / (np.sqrt(r_hat) + self.error)
        return self.learning_rate * adjusted_gradient, r, s
    
    def train_step(self,layers_grad,layers):
        for (W,b), (W_g,b_g), (W_r,b_r), (W_s,b_s) in zip(layers, layers_grad, self.r, self.s): 
            W_i, W_r, W_s = self.training_increment(W_g, W_r, W_s)
            b_i, b_r, b_s = self.training_increment(b_g, b_r, b_s)
            W -= W_i
            b -= b_i
        return layers 