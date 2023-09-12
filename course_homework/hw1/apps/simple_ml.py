import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filename) as f:
      # First 16 bytes are magic number, number of images, row and col,
      # and the data is from 16 offset in unsigned byte.
      # So read from 16 offset by uint8.
      pixels = np.frombuffer(f.read(), dtype='uint8', offset=16)
      # Reshape to 784 and normalizate to 0.0 ~ 1.0
      pixels = pixels.reshape(-1, 28 * 28).astype('float32') / 255 
    # load lables
    with gzip.open(label_filename) as f:
      # First 8 bytes are magic_number, number of labels.
      # Read from 8 offset by uint8
      uint8_lables = np.frombuffer(f.read(), dtype = 'uint8', offset = 8)
    
    return (pixels, uint8_lables)


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    Z_y = ndl.ops.summation(Z * y_one_hot)
    Z_sum = ndl.ops.summation(ndl.ops.log(ndl.ops.summation(ndl.ops.exp(Z), axes = (1, ))))
    return (Z_sum - Z_y) / Z.shape[0]


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (ndl.Tensor[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    num_examples = X.shape[0]
    for i in range(num_examples // batch + 1):
        j = min((i+1) * batch, num_examples) # Up bound in this batch
        X_b = ndl.Tensor(X[i*batch : j]) # train data, (num_examples x input_dim)
        y_b = y[i*batch : j] # train lables
        m = j - i*batch
        if(m == 0):
            break;
        
        # Compute result
        H = X_b @ theta # (num_examples x input_dim)·(input_dim x num_classes)

        # Compute normalization
        H_exp = ndl.ops.exp(H) # (num_examples x num_classes)
        exp_sum = H_exp.sum(axes = 1) # (num_examples)
        Z = H_exp / exp_sum.reshape(m, 1)

        # Compute I
        I_ = np.zeros_like(Z)
        I_[np.arange(m), y_b] = 1
        I = ndl.Tensor(I_)
        
        # Compute grad
        grad = transpose(X_b) @ (Z - I) / m

        # Update theta
        theta -= lr * grad


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    num_examples = X.shape[0]
    for i in range(num_examples // batch + 1):
        j = min((i+1) * batch, num_examples)
        X_b = ndl.Tensor(X[i * batch : j])
        y_b = y[i * batch : j]
        m = j - i * batch
        if(m == 0):
            break
        
        # Compute forward
        Z1 = ndl.ops.relu(X_b @ W1) # (num_examples x input_dim)·(input_dim, hidden_dim)
        Z2 = Z1 @ W2 # (num_examples x hidden_dim)·(hidden_dim, num_classes)
        
        # Compute the loss
        y_one_hot = np.zeros(Z2.shape, dtype = "float32")
        y_one_hot[np.arange(Z2.shape[0]), y_b] = 1
        y_one_hot = ndl.Tensor(y_one_hot, requires_grad = False)
        loss = softmax_loss(Z2, y_one_hot)
        
        # Gradient backward
        loss.backward()

        # Update the parameters
        W1.data -= lr * ndl.Tensor(W1.grad.numpy().astype(np.float32)) 
        W2.data -= lr * ndl.Tensor(W2.grad.numpy().astype(np.float32))
    
    return W1, W2
        

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
