from __future__ import annotations
from abc import ABC, abstractmethod

# Typing
from .typing_utils import ArrayF, NetworkParams
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np  # typed NumPy for the checker
    from .training import TrainingMethod
else:
    import autograd.numpy as np  # runtime


class StepMethod(ABC):
    """Abstract base class for step methods used in in the training loop."""

    caller: "TrainingMethod"
    learning_rate: float
    @abstractmethod
    def setup(self, starting_layers: NetworkParams) -> None:
        """Setup any necessary internal state before training begins."""
        ...

    @abstractmethod
    def train_step(self, layers_grad: NetworkParams, layers: NetworkParams) -> NetworkParams: 
        """Performs a single training step, updating the layers in place."""
        ...



# ========== Step methods ==========

class ConstantLearningRateStep(StepMethod):
    """Simple gradient descent with a fixed learning rate. Scales gradients by a constant factor and subtracts from parameters."""
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    def training_increment(self, gradient: ArrayF):
        return self.learning_rate * gradient
    
    def train_step(self, layers_grad: NetworkParams, layers: NetworkParams):
        for (W,b), (W_g,b_g) in zip(layers, layers_grad): 
            W -= self.training_increment(W_g)
            b -= self.training_increment(b_g)
        return layers

class MomentumStep(StepMethod):
    """Gradient descent with momentum. Accumulates a velocity term to damp oscillations."""
    def __init__(self, learning_rate: float, momentum: float) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def setup(self, starting_layers: NetworkParams):
        # Initialize velocity array with same shapes as network parameters
        self.velocity: NetworkParams = []
        for (W,b) in starting_layers:
            self.velocity.append((np.zeros_like(W), np.zeros_like(b)))
    
    def training_increment(self, gradient: ArrayF, velocity: ArrayF):
        velocity = self.momentum * velocity + self.learning_rate * gradient
        return velocity
    
    def train_step(self, layers_grad: NetworkParams, layers: NetworkParams):
        for (W,b), (W_g,b_g), (W_vel,b_vel) in zip(layers, layers_grad, self.velocity): 
            W_vel = self.training_increment(W_g,W_vel)
            b_vel = self.training_increment(b_g,b_vel)
            W -= W_vel
            b -= b_vel
        return layers


class ADAgradStep(StepMethod):
    """ADAgrad optimization algorithm. Adapts learning rate for each parameter based on historical gradients."""
    def __init__(self, learning_rate: float, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.error = error
    
    def setup(self, starting_layers: NetworkParams):
        self.accumulated_gradient: NetworkParams = []
        for (W,b) in starting_layers:
            self.accumulated_gradient.append((np.zeros_like(W), np.zeros_like(b)))
        
    def training_increment(self, gradient: ArrayF, accumulated_gradient: ArrayF):
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
    """RMSprop optimization algorithm. Uses a weighted average of squared gradients."""
    def __init__(self, learning_rate: float, decay_rate: float, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.error = error
        
    def setup(self, starting_layers: NetworkParams) -> None:
        # Set up accumulated gradient array with same shapes as network parameters
        self.accumulated_gradient: NetworkParams = []
        for (W,b) in starting_layers:
            self.accumulated_gradient.append((np.zeros_like(W), np.zeros_like(b)))
    
    def training_increment(self, gradient: ArrayF, accumulated_gradient: ArrayF):
        accumulated_gradient = self.decay_rate * accumulated_gradient + (1 - self.decay_rate) * gradient**2
        adjusted_gradient = gradient / (np.sqrt(accumulated_gradient) + self.error)
        return self.learning_rate * adjusted_gradient, accumulated_gradient
    
    def train_step(self, layers_grad: NetworkParams, layers: NetworkParams):
        for (W,b),(W_g,b_g),(W_g_acc,b_g_acc) in zip(layers, layers_grad, self.accumulated_gradient): 
            W_i, W_g_acc = self.training_increment(W_g, W_g_acc)
            b_i, b_g_acc = self.training_increment(b_g, b_g_acc)
            W -= W_i
            b -= b_i
        return layers       
    
class AdamStep(StepMethod):
    """Adam optimization algorithm. Combines momentum and RMSprop ideas."""
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, error: float = 1e-8) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.error = error

    def setup(self, starting_layers: NetworkParams) -> None:
        self.t = 0  # Time step
        self.s: NetworkParams = []   # First moment vector
        self.r: NetworkParams = []   # Second moment vector
        for (W,b) in starting_layers:
            self.s.append((np.zeros_like(W), np.zeros_like(b)))
            self.r.append((np.zeros_like(W), np.zeros_like(b)))
    
    def training_increment(self, gradient: ArrayF, r: ArrayF, s: ArrayF):
        self.t += 1

        s = self.beta1 * s + (1 - self.beta1) * gradient
        r = self.beta2 * r + (1 - self.beta2) * (gradient ** 2)

        s_hat = s / (1 - self.beta1 ** self.t)
        r_hat = r / (1 - self.beta2 ** self.t)

        adjusted_gradient = s_hat / (np.sqrt(r_hat) + self.error) # Include a small error to avoid division by 0
        return self.learning_rate * adjusted_gradient, r, s
    
    def train_step(self, layers_grad: NetworkParams, layers: NetworkParams):
        for (W,b), (W_g,b_g), (W_r,b_r), (W_s,b_s) in zip(layers, layers_grad, self.r, self.s): 
            W_i, W_r, W_s = self.training_increment(W_g, W_r, W_s)
            b_i, b_r, b_s = self.training_increment(b_g, b_r, b_s)
            W -= W_i
            b -= b_i
        return layers 