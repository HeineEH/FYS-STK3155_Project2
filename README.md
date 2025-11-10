# FYS-STK3155 Project 2

## Group members

Frederik Callin Østern, Heine Elias Husdal

## Project description

In this project, we investigate feed forward neural networks. In particular, we do a regression analysis on data taken from the one dimensional Runge function and a classification analysis on the MNISt dataset. In the case of regression, we investigate different training methods, including stochastic gradient descent (SGD) and ordinary gradient descent (GD), with both the Adam and RMSprop algorithms. We optimize the model architecture (given by the number of hidden layers and the number of nodes, as well as the activation functions used) in order to minimize the cost function. Moreover, we investigate whether $L_1$ or $L_2$ regularization improves model performance. With our analysis, one gains a better understanding of the performance of neural networks, as well as how they should be trained.

## Folder structure

`code/` - Folder with all the code files, including:

- `utils/` Folder with all the code python files with the methods/functions, including:
  - `activation_functions.py` Code implementing the activation functions
  - `cost_functions.py` Code implementing the cost functions
  - `neural_network.py` Code implementing the neural network
  - `step_methods.py` Code implementing the various GD algorithms, including constant step, momentum, Adagrad, RMSprop and Adam
  - `training.py` Code for training the neural network, including with either stochastic gradient descent or ordinary gradient descent
  - `typing_utils.py` Code implementing some type hints
  - `utils.py` Code implementing various extra useful functions
- `exercise_2b.ipynb` Notebook solving exercise 2b
- `exercise_2d.ipynb` Notebook solving exercise 2d
- `exercise_2e.ipynb` Notebook solving exercise 2e
- `test_animation.ipynb` Notebook for animating the improvement of the approximation to the Runge function during training of the neural network
- `test_autograd.ipynb` Notebook for checking the consistency between the gradients due to autograd and our manual backpropagation implementation
- `test.ipynb` Notebook with a simple test of the neural network class

## Running the code

1. Set up a python virtual environment

2. Install packages: `pip install -r requirements.txt`

3.
