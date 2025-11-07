from __future__ import annotations
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from utils.neural_network import NeuralNetwork
from utils.training import TrainingMethod

# Typing
from .typing_utils import ArrayF, NetworkParams
from typing import TYPE_CHECKING, TypeVar
if TYPE_CHECKING:
    import numpy as np  # typed NumPy for the checker
else:
    import autograd.numpy as np  # runtime


T = TypeVar("T", float, ArrayF)
def runge(x: T) -> T:
    return 1./(1. + 25*x**2)

def generate_dataset(num = 400, noise = 0.05):
    x = np.random.uniform(-1, 1, num).reshape(-1, 1)
    y = runge(x) + noise*np.random.normal(0, 1, (num, 1))
    return x, y

def onehot_encode_mnist_labels(y):
    return np.eye(10)[y.astype(int)]

def onehot_decode_mnist_labels(y_onehot):
    return np.argmax(y_onehot, axis=1)

def get_MNIST_dataset():
    # Fetch the MNIST dataset

    # Fetch the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    # Extract data (features) and target (labels)
    X = mnist.data
    y = mnist.target

    # Scale the data to [0, 1]
    X = X / 255.0

    # One-hot encode the labels
    y = onehot_encode_mnist_labels(y)

    return X, y

def show_images(X, y):
    cols = 5
    rows = int(len(X)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for image, title_text in zip(X, y):        
        plt.subplot(rows, cols, index)        
        plt.imshow(image.reshape(28, 28), cmap="gray")
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

def check_layer_params_equality(params1: NetworkParams, params2: NetworkParams) -> bool:
    if len(params1) != len(params2):
        raise ValueError("params1 and params2 must have the same number of layers")
    
    for (p1_W, p1_b), (p2_W, p2_b) in zip(params1, params2):
        if p1_W.shape != p2_W.shape or p1_b.shape != p2_b.shape:
            raise ValueError("Layer parameter shapes do not match")

        if not (np.allclose(p1_W, p2_W) and np.allclose(p1_b, p2_b)):
            return False
        
    return True

def plot_mse_data(mse_data: ArrayF):
    if mse_data.shape[1] != 3:
        raise ValueError("mse_data must have shape (num_iterations, 3): (iterations, train_mse, test_mse)")
    plt.plot(mse_data[:, 0], mse_data[:, 1], label="Train MSE", alpha=0.8)
    plt.plot(mse_data[:, 0], mse_data[:, 2], label="Test MSE")
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.legend()


def analyze_model_learning_rates(
        model: NeuralNetwork,
        training_method: TrainingMethod,
        learning_rates: list[float],
        num_iterations = 3000, 
        track_mse = False
    ):
    if track_mse:
        mse_data = np.zeros((num_iterations,len(learning_rates)))
    else: 
        mse_data = np.zeros((len(learning_rates)))
    for i, learning_rate in enumerate(learning_rates):
        model.reset_layers(random_state=124)
        training_method.step_method.learning_rate = learning_rate
        mse_vals = model.train(training_method, num_iterations, n_batches=5,track_mse = track_mse)
        if (training_method.test_inputs is not None) and (training_method.test_targets is not None):
            if track_mse: 
               mse_data[:,i] = mse_vals[:,2] 
            else: 
                mse_data[i] = model.mse_batch(training_method.test_inputs, training_method.test_targets)
            
    return mse_data


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm*100)
    disp.plot(cmap='Blues', values_format='.1f', )
    plt.gcf().set_size_inches(7, 6)

    return cm