# Week 5 Assignment
These are the assets related to the week 5 assignment, `FullyConnectedNets.ipynb`

Note: the notebook listed here is **not** Google Colab optimized, but can be adapted for Colab. Do the following, then upload to Colab (make sure to adhere to the specified file structure)
1. adding a cell at the top of the notebook with the following code block, and
2. replacing `None` with your Google Drive's path to the `cs231n` folder (as specified in the comment block)

```
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

"""
# enter the foldername in your Drive where you have saved the unzipped
# 'cs231n' folder containing the '.py', 'classifiers' and 'datasets'
# folders.
# e.g. 'cs231n/assignments/assignment2/cs231n/'
"""
FOLDERNAME = None

assert FOLDERNAME is not None, "[!] Enter the foldername."

%cd drive/My\ Drive
%cp -r $FOLDERNAME ../../
%cd ../../
%cd cs231n/datasets/
!bash get_datasets.sh
%cd ../../
```

## File structure
* `assignment2/`
    * **`FullyConnectedNets.ipynb`**
    * `cs231n/`
        * **`layers.py`**
        * **`optim.py`**
        * **`solver.py`**
        * `classifiers/`
            * **`fc_net.py`**

## Context and meta
* **`FullyConnectedNets.ipynb`**
    * Practice implementing various optimizers, as well as flexible/reusable forward prop and backprop layer implementations, leading up to multi-layered affine nets of arbitrary (one or more hidden layers) depth
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