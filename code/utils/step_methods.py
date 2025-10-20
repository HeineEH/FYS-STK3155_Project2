from __future__ import annotations
import numpy as nup
import numpy.typing as npt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from training import _TrainingMethod

# Template for step methods, like gd-momentum, RMSprop, ADAgrad
class _StepMethod:
    caller: "_TrainingMethod"
    learning_rate: float
    def setup(self, starting_layers) -> None:
        ...
    def training_increment(self, gradient: npt.NDArray[nup.floating]) -> None:
        ...



# ========== Step methods ==========

class ConstantLearningRateStep(_StepMethod):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    def training_increment(self, gradient: npt.NDArray[nup.floating]) -> None:
        return self.learning_rate * gradient
    
    def train_step(self,layers_grad,layers):
        for (W,b),(W_g,b_g) in zip(layers,layers_grad): 
            W -= self.training_increment(W_g)
            b -= self.training_increment(b_g)
        return layers

class MomentumStep(_StepMethod):
    def __init__(self, learning_rate: float, momentum: float, ) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def setup(self, starting_layers) -> None:
        self.velocity = []
        for (W,b) in starting_layers: 
            self.velocity.append((nup.zeros_like(W),nup.zeros_like(b)))
    
    def training_increment(self, gradient: npt.NDArray[nup.floating],velocity) -> None:
        velocity = self.momentum * velocity + self.learning_rate * gradient
        return velocity
    
    def train_step(self,layers_grad,layers):
        for (W,b),(W_g,b_g),(W_vel,b_vel) in zip(layers,layers_grad,self.velocity): 
            W_vel = self.training_increment(W_g,W_vel)
            b_vel = self.training_increment(b_g,b_vel)
            W -= W_vel
            b -= b_vel
        return layers


class ADAgradStep(_StepMethod):
    def __init__(self, learning_rate: float, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.error = error
    
    def setup(self, starting_layers) -> None:
        self.accumulated_gradient = []
        for (W,b) in starting_layers: 
            self.accumulated_gradient.append((nup.zeros_like(W),nup.zeros_like(b)))
        
    def training_increment(self, gradient: npt.NDArray[nup.floating],accumulated_gradient: npt.NDArray[nup.floating]) -> None:
        accumulated_gradient += gradient**2  # Accumulate squared gradients
        adjusted_gradient = gradient / (nup.sqrt(accumulated_gradient) + self.error)
        return self.learning_rate * adjusted_gradient, accumulated_gradient
    
    def train_step(self,layers_grad,layers):
        for (W,b),(W_g,b_g),(W_g_acc,b_g_acc) in zip(layers,layers_grad,self.accumulated_gradient): 
            W_i, W_g_acc = self.training_increment(W_g,W_g_acc)
            b_i, b_g_acc = self.training_increment(b_g,b_g_acc)
            W -= W_i
            b -= b_i
        return layers

class RMSpropStep(_StepMethod):
    def __init__(self, learning_rate: float, decay_rate: float, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.error = error
        
    def setup(self, starting_layers) -> None:
        self.accumulated_gradient = []
        for (W,b) in starting_layers: 
            self.accumulated_gradient.append((nup.zeros_like(W),nup.zeros_like(b)))
    
    def training_increment(self, gradient: npt.NDArray[nup.floating],accumulated_gradient: npt.NDArray[nup.floating]) -> None:
        accumulated_gradient = self.decay_rate * accumulated_gradient + (1 - self.decay_rate) * gradient**2
        adjusted_gradient = gradient / (nup.sqrt(accumulated_gradient) + self.error)
        return self.learning_rate * adjusted_gradient, accumulated_gradient
    
    def train_step(self,layers_grad,layers):
        for (W,b),(W_g,b_g),(W_g_acc,b_g_acc) in zip(layers,layers_grad,self.accumulated_gradient): 
            W_i, W_g_acc = self.training_increment(W_g,W_g_acc)
            b_i, b_g_acc = self.training_increment(b_g,b_g_acc)
            W -= W_i
            b -= b_i
        return layers       
    
class AdamStep(_StepMethod):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.error = error

    def setup(self, starting_layers) -> None:
        self.t = 0  # Time step
        self.s = []   # First moment vector
        self.r = []   # Second moment vector
        for (W,b) in starting_layers: 
            self.s.append((nup.zeros_like(W),nup.zeros_like(b)))
            self.r.append((nup.zeros_like(W),nup.zeros_like(b)))
    
    def training_increment(self, gradient: npt.NDArray[nup.floating], r: npt.NDArray[nup.floating], s: npt.NDArray[nup.floating]) -> None:
        self.t += 1

        s = self.beta1 * s + (1 - self.beta1) * gradient
        r = self.beta2 * r + (1 - self.beta2) * (gradient ** 2)

        s_hat = s / (1 - self.beta1 ** self.t)
        r_hat = r / (1 - self.beta2 ** self.t)

        adjusted_gradient = s_hat / (nup.sqrt(r_hat) + self.error)
        return self.learning_rate * adjusted_gradient, r, s
    
    def train_step(self,layers_grad,layers):
        for (W,b),(W_g,b_g),(W_r,b_r),(W_s,b_s) in zip(layers,layers_grad,self.r,self.s): 
            W_i, W_r,W_s = self.training_increment(W_g,W_r,W_s)
            b_i, b_r,b_s = self.training_increment(b_g,b_r,b_s)
            W -= W_i
            b -= b_i
        return layers 