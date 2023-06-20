# MNIST-neural-network

This C++ program is a neural network classifier for the MNIST handwritten digit dataset (N = 60,000). OpenCV is used to read the images. Armadillo is used for optimised matrix multiplication (BLAS).

The labelled dataset is paritioned into a training set (n = 50,0000) and a test set (n = 10,000). The network uses a quadratic cost function, the sigmoid activation function, and Xavier-initialised weights. Mini-batch gradient descent is used for training. Following the completion of a training epoch, the training set is shuffled and the network performance is evaluated by the fraction of the test set accurately classified.

Typically, the network reaches around 96% accuracy on the test set within 10-20 epochs of training with a mini-batch size of 10 and a learning rate of 0.3 (On an M1 MacBook Pro, built for release with -O3, such an epoch takes ~4s).

To test yourself, install the OpenCV and Amradillo libraries and configure your compiler accordingly. Filepaths for the MNIST images and labels files can be specified in main.cpp.

