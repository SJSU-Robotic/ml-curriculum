# ML Track | Module 2
Navigate to [README](README.md) | [Module 1](module1.md) ⏮ Module 2 ⏭ [Module 3](module3.md)

## Linear layers in neural networks
This set of tutorials follow on the heels of the 60-minute blitz's neural networks tutorial. Previously, you were given the 2-Conv2D+3-Linear network in `class Net()`. In this module, you'll get to explore some of the underlying mechanism that goes into implementing `torch.nn`. Specifically, we'll be exploring the standard linear layer - `nn.Linear`.

First, start with [What is `torch.nn` really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html). When they start talking about convolutional neural networks - stop! We'll get into that later on.

For now, try replacing the model's single layer in `Mnist_Logistic()` with the multi-layer perceptron presented [in this example](https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a).

Compare the code you have with the code presented [here](https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210) - does the code make sense to you? Things to look out for:
* How is training data loaded?
* How is the model being defined?
* How is the training loop written?

Let's look at more examples of this in the [next module](module3.md) as we explore convolutional neural networks.