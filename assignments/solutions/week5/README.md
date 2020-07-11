# Week 5 Assignment
These are the assets related to the week 5 assignment, `FullyConnectedNets.ipynb`

## File structure
* `assignment2`
    * **`FullyConnectedNets.ipynb`**
    * `cs231n`
        * **`layers.py`**
        * **`optim.py`**
        * **`solver.py`**
        * `classifiers`
            * **`fc_net.py`**

## Context and meta
* **`FullyConnectedNets.ipynb`**
    * Jupyter notebook for the assignment 
* **`layers.py`**
    * Assigned TODOs/tasks to students - implementation of activation/propagations for
        * `affine_forward()`
        * `affine_backward()`
        * `relu_forward()`
        * `relu_backward()`
* **`optim.py`**
    * Assigned TODOs/tasks to students - implementation of optimizers for
        * `sgd_momentum()`
        * `rmsprop()`
        * `adam()`
* **`solver.py`**
    * Added `verbose_epoch` flag to toggle epoch verbosity and separate it from training iteration verbosity. This modifies the solver/trainer's logging behavior
* **`fc_net.py`**
    * Assigned TODOs/tasks to students - implementation of fully-connected/affine/dense nets for
        * `TwoLayerNet`: a two-layer affine net
            * `__init__()`
            * `loss()`
        * `FullyConnectedNet`: an affine net of arbitrary (one or more hidden layers) depth
            * `__init__()`
            * `loss()`