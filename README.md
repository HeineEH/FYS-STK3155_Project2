# FYS-STK3155 Project 2

## Group members

Frederik Callin Østern, Heine Elias Husdal

## Project description

In this project, we investigate feed forward neural networks. In particular, we perform a regression analysis on data taken from the one dimensional Runge function and a classification analysis on the MNIST dataset.

In the case of regression, we investigate different training methods, including stochastic gradient descent (SGD) and ordinary gradient descent (GD), with both the Adam and RMSprop algorithms. We optimize the model architecture (given by the number of hidden layers and the number of nodes, as well as the activation functions used) in order to minimize the cost function. Moreover, we investigate whether $L_1$ or $L_2$ regularization improves model performance. With our analysis, one gains a better understanding of the performance of neural networks, as well as how they should be trained.

In the classification part of the project, we apply feed-forward neural networks (FFNNs) to the MNIST dataset of handwritten digits. The network is trained using the Adam optimizer with multiclass cross-entropy loss. We perform a grid search over key hyperparameters, including the learning rate, regularization strength, number of hidden layers, number of nodes per layer, and activation functions. The results show how these parameters affect both training efficiency and test-set accuracy. Finally, we compare our own implementation with a corresponding _PyTorch_ model to verify implementation consistency and performance.

## Folder structure

`code/` - Folder with all the code files, including:

- `exercise_2b.ipynb` - Runge regression using our own implementation of a feed forward neural network
- `exercise_2c.ipynb` - PyTorch comparison of Runge regression in exercise **2b**
- `exercise_2d.ipynb` - Testing different activation functions and depths of the neural network
- `exercise_2e.ipynb` - Testing different norms
- `exercise_2f.ipynb` - MNIST classification using our own implementation of a feed forward neural network, with hyperparameter tuning
- `exercise_2f_pytorch.ipynb` - PyTorch comparison of MNIST classification in exercise **2f**

- `utils/` - Reusable code for the neural network implementation, including:
  - `activation_functions.py` - Activation function classes (Sigmoid, ReLU, Softmax, ...)
  - `cost_functions.py` - Cost function classes with regularization implemented (MSE, Cross-Entropy, ...)
  - `neural_network.py` - Handwritten feedforward neural network, implementing forward- and backpropagation.
  - `step_methods.py` - Gradient-step algorithms (Adam, RMSprop, AdaGrad, Momentum, ...)
  - `training.py` - Code for training the neural network, including stochastic GD and plain GD
  - `typing_utils.py` - Type aliases used throughout the project
  - `utils.py` - Code implementing various extra useful functions (data loading, plotting, ...)

Notes

- The notebooks in `code/` reproduce the experiments and figures in the report.
- Most notebooks take a while to run, especially those involving hyperparameter searches. To load our computed results directly, set the top variable `LOAD_FROM_FILE = True` in each of the notebooks.
- The core implementation is under `code/utils/` and is importable from the notebooks.

## Running the code

1. Set up a python virtual environment

2. Install packages: `pip install -r requirements.txt`

3. Run `.ipynb` notebooks described above.
