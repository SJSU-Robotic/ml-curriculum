# Week 7 Assignment
These are the assets related to the week 7 assignments, `Dropout.ipynb` and `ConvolutionalNetworks.ipynb`

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
    * **`Dropout.ipynb`**
    * **`ConvolutionalNetworks.ipynb`**
    * `cs231n/`
        * **`layers.py`**
        * `classifiers/`
            * **`fc_net.py`**
            * **`cnn.py`**

## Context and meta
* **`Dropout.ipynb`**
    * Practice implementing the dropout layer and explore its behavior
* **`ConvolutionalNetworks.ipynb`**
    * Practice implementing and explore the behaviors of convolutional layers, max pooling layers, and CNNs
    * Practice implementing and explore the behaviors of spatial batch normalization and spatial group normalization
* **`layers.py`**
    * Contains assets for both assignments. Revert to commit `0b13bd3`, for a `Dropout.ipynb`-specific `layers.py`
    * Assigned TODOs/tasks to students - implementation of activation/propagations for
        * `dropout_forward()`
        * `dropout_backward()`
        * `conv_forward_naive()`
        * `conv_backward_naive()`
        * `max_pool_forward_naive()`
        * `max_pool_backward_naive()`
        * `spatial_batchnorm_forward()`
        * `spatial_batchnorm_backward()`
        * `spatial_groupnorm_forward()`
        * `spatial_groupnorm_backward()`
* **`fc_net.py`**
    * Assigned TODOs/tasks to students - dropout enhancements to the implementation of fully-connected/affine/dense nets for
        * `FullyConnectedNet`: an affine net of arbitrary (one or more hidden layers) depth
            * `__init__()`
            * `loss()`
* **`cnn.py`**
    * Assigned TODOs/tasks to students - implementation of `ThreeLayerConvNet`
        * `ThreeLayerConvNet`: a CNN with one conv/relu/pool layer, an affine/relu layer, and a simple affine layer
            * `__init__()`
            * `loss()`