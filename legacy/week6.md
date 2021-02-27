# ML Track | Week 6
[Week 5](week5.md) ⬅️ Week 6 ➡️ [Week 7](week7.md)

## Readings

- [Lecture 7 Slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf)
- [(Repeated from last week) Training Neural Nets 3](http://cs231n.github.io/neural-networks-3/))

## Lecture Videos

- [Lecture 7 | Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7)

## Homework
* [Practice PyTorch](https://github.com/gauravkuppa/pytorch-examples)
  * Read through the examples
  * Compare our implementations of concepts from weeks 1-5 to PyTorch
    * How are inputs, outputs, weights, and biases represented? 
      * Where do the weights and biases go when we start using `torch.nn`?
    * How is loss, forward prop, and backward prop handled?
      * How are the gradients generated?
    * What hyperparameters are presented in these examples?
      * How are these hyperparameters handled?
  * Notes
    * Since the introduction of [TensorFlow Fold](https://ai.googleblog.com/2017/02/announcing-tensorflow-fold-deep.html), TensorFlow also offers dynamic computational graphs
    * In case you are curious about alternatives such as Keras and TensorFlow, take a look at [this article](https://towardsdatascience.com/keras-vs-pytorch-for-deep-learning-a013cb63870d).

## Review

- [Implement batch normalization](assignments/colab/2020/assignment2/BatchNormalization.ipynb)
    * [Solution](assignments/solutions/week6/README.md)
    