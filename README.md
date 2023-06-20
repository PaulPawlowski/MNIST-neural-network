# MNIST-neural-network

This C++ program is a neural network classifier for the MNIST handwritten digit dataset. The dataset can be downloaded from http://yann.lecun.com/exdb/mnist/. OpenCV is used to read the images. Armadillo is used for faster linear algebra computation (BLAS).

The dataset (n = 70,000) is comprised of a training set (n = 60,0000) and a test set (n = 10,000). The network uses a quadratic cost function, sigmoid activation functions, and Xavier-initialised weights. Mini-batch gradient descent is used for training. Following the completion of a training epoch, the training set is shuffled and the network's performance is measured by evaluating the fraction of the test set accurately classified.

Typically, the network reaches around 96% accuracy on the test set within 10-20 epochs of training with a mini-batch size of 10 and a learning rate of 0.3 (On an M1 MacBook Pro, built for release with -O3, such an epoch takes ~4s).

To test yourself, install the OpenCV and Amradillo libraries and configure your compiler accordingly. Filepaths for the MNIST images and labels files are specified in main.cpp.

