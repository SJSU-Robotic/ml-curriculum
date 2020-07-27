# Week 5 Assignment
These are the assets related to the week 6 assignment, `BatchNormalization.ipynb`

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
    * **`BatchNormalization.ipynb`**
    * `cs231n/`
        * **`layers.py`**
        * `classifiers/`
            * **`fc_net.py`**

## Context and meta
* **`BatchNormalization.ipynb`**
    * Practice implementing batch normalization and layer normalization, and explore its behavior
* **`layers.py`**
    * Assigned TODOs/tasks to students - implementation of activation/propagations for
        * `batchnorm_forward()`
        * `batchnorm_backward()`
        * `batchnorm_backward_alt()`
        * `layernorm_forward()`
        * `layernorm_backward()`
* **`fc_net.py`**
    * Assigned TODOs/tasks to students - normalization enhancements to the implementation of fully-connected/affine/dense nets for
        * `FullyConnectedNet`: an affine net of arbitrary (one or more hidden layers) depth
            * `__init__()`
            * `loss()`
            * `affine_batchnorm_forward()`
            * `affine_batchnorm_backward()`
            * `affine_layernorm_forward()`
            * `affine_layernorm_backward()`