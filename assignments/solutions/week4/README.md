# Week 4 Assignment
These are the assets related to the week 4 assignments, `two_layer_net.ipynb` and `features.ipynb`

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
# e.g. 'cs231n/assignments/assignment1/cs231n/'
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
* `assignment1/`
    * **`two_layer_net.ipynb`**
    * **`features.ipynb`**
    * `cs231n/`
        * `classifiers/`
            * **`neural_net.py`**

## Context and meta
* **`two_layer_net.ipynb`**
* **`features.py`**
* **`neural_net.py`**