import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    # Comstruct module for residual block
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),

        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    residualblock = nn.Sequential(
        nn.Residual(modules),
        nn.ReLU()
    )
    return residualblock


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    modules = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]

    for i in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)


def epoch(dataloader, model, opt=None):
    """
    @return : (average error rate (1 - accuracy) (as a float), 
                the average loss over all samples (as a float))
    """
    np.random.seed(4)
    loss_func = nn.SoftmaxLoss()
    correct, loss_sum, n_step, n_samples = 0., 0., 0, 0

    # Set the status tarin or eval
    if opt:
        model.train()
    else:
        model.eval()
    
    for X, y in dataloader:
        # reset the grad param in every batch
        forward = model(X)
        loss = loss_func(forward, y)
        if opt:
            # Clear last grad
            opt.reset_grad()
            # autograd
            loss.backward()
            # update the params
            opt.step()

        correct += (forward.numpy().argmax(axis = 1) == y.numpy()).sum()
        loss_sum += loss.numpy()
        n_step += 1
        n_samples += X.shape[0]

    return (1 - correct / n_samples), loss_sum / n_step


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    train_data = ndl.data.MNISTDataset(
        data_dir + '/train-images-idx3-ubyte.gz',
        data_dir + '/train-labels-idx1-ubyte.gz'
    )
    test_data = ndl.data.MNISTDataset(
        data_dir + '/t10k-images-idx3-ubyte.gz',
        data_dir + '/t10k-labels-idx1-ubyte.gz',
    )
    train_loader = ndl.data.DataLoader(train_data, batch_size)
    test_loader = ndl.data.DataLoader(test_data, batch_size)

    model = MLPResNet(28 * 28, hidden_dim)
    opt = optimizer(model.parameters(), lr = lr, weight_decay = weight_decay)
    for _ in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        print()
        print("train_acc = %.2f train_loss = %.2f" % (train_acc, train_loss))
        test_acc, test_loss = epoch(test_loader, model)
        print("test_acc = %.2f test_loss = %.2f" % (test_acc, test_loss))

    return (train_acc, train_loss, test_acc, test_loss)

if __name__ == "__main__":
    train_mnist(data_dir="../data")
    print("ending")
