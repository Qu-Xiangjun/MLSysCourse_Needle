import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


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
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    
    # load image
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
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # Get the correct simge classifies for output value.
    # y indicates the correct classify index.
    # np.arange(Z.shape[0]) is a iterator for every row of Z.
    Z_y = Z[np.arange(Z.shape[0]), y]

    # Get log for sum of exp(Zi)
    Z_sum = np.log(np.exp(Z).sum(axis = 1))

    # return the mean Softmax loss
    return np.mean(Z_sum - Z_y)
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    for i in range(num_examples // batch + 1):
      j = min((i+1) * batch, num_examples) # Up bound in this batch
      X_b = X[i*batch : j] # train data, (num_examples x input_dim)
      y_b = y[i*batch : j] # train lables
      m = j - i*batch
      if(m == 0):
        break;
      
      # Compute result
      H = X_b.dot(theta) # (num_examples x input_dim)·(input_dim x num_classes)

      # Compute normalization
      H_exp = np.exp(H) # (num_examples x num_classes)
      exp_sum = H_exp.sum(axis = 1) # (num_examples)
      Z = H_exp / exp_sum.reshape(m, 1)

      # Compute I
      I = np.zeros_like(Z)
      I[np.arange(m), y_b] = 1
      
      # Compute grad
      grad = X_b.T.dot(Z - I) / m

      # Update theta
      theta -= lr * grad
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    for i in range(num_examples // batch + 1):
      j = min((i+1) * batch, num_examples)
      X_b = X[i * batch : j]
      y_b = y[i * batch : j]
      m = j - i * batch
      if(m == 0):
        break
      
      # Compute forward
      Z1 = np.maximum(X_b.dot(W1), 0) # (num_examples x input_dim)·(input_dim, hidden_dim)
      Z2 = Z1.dot(W2) # (num_examples x hidden_dim)·(hidden_dim, num_classes)

      # Compute grad for W2
      Z2_exp = np.exp(Z2) # (num_examples x num_classes)
      Z2_normalize = Z2_exp / Z2_exp.sum(axis = 1, keepdims = True)
      I2 = np.zeros_like(Z2_normalize)
      I2[np.arange(m), y_b] = 1
      G2 = Z2_normalize - I2
      grad_W2 = Z1.T.dot(G2) / m # (num_examples x hidden_dim).T·(num_examples x num_classes)

      # Compute grad for W1
      I1 = np.zeros_like(Z1)
      I1[Z1 > 0] = 1
      G1 = G2.dot(W2.T) * I1 # (num_examples x num_classes)· (hidden_dim, num_classes).T
      grad_W1 = X_b.T.dot(G1) / m # (num_examples x input_dim).T·(num_examples x hidden_dim)

      # update grad
      W1 -= lr * grad_W1
      W2 -= lr * grad_W2
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
