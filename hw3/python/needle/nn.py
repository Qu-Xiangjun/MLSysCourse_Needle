"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    """Gets the parameters of a given value recursively, which can include
    parameter, module, dict , list and tuple."""
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    """Gets the child module list of the given NN Module recursively."""
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    # NOTE: self.__dict__ is the symbol table, which contact all the class member. 
    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Init weight and bias by kaiming_uniform
        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features, requires_grad = True
        ))
        if bias:
            # using reshape because the kaiming compute conly by param for fan_in
            self.bias = Parameter(init.kaiming_uniform(
                out_features, 1, requires_grad = True
            ).reshape((1, out_features)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        X_mul_weight = X @ self.weight
        if self.bias:
            return X_mul_weight + self.bias.broadcast_to(X_mul_weight.shape)
        return X_mul_weight


class Flatten(Module):
    """Reshape the X shape (B, x0, x1, ...) to shape (B, x0*x1*...), which make
    the non-batch dimention is flatten""" 
    def forward(self, X):
        return X.reshape((X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    """Compute a seqential module with first module's inputs and return the last
    module's outputs."""
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        """ Return softmax loss.  
        Args:
            logits (ndl.Tensor[np.float32]): 2D Tensor of shape
                (batch_size, num_classes), containing the logit predictions for
                each class.
            y (ndl.Tensor[np.int8]): 1D Tensor of shape (batch_size)
                containing the index of the true label of each example and
                zeros elsewhere.

        Returns:
            Average softmax loss over the sample. (ndl.Tensor[np.float32])
        """
        exp_sum = ops.logsumexp(logits, axes=(1, )).sum()
        z_y = (logits * init.one_hot(logits.shape[1], y)).sum()
        return (exp_sum - z_y) / logits.shape[0]



class BatchNorm1d(Module):
    """Applies batch normalization over a mini-batch of inputs as described 
    in the paper 'Batch Normalization: Accelerating Deep Network Training by 
    Reducing Internal Covariate Shift(https://arxiv.org/abs/1502.03167)'."""
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim 
        self.eps = eps
        self.momentum = momentum
        # Init the parameter
        self.weight = Parameter(init.ones(self.dim, requires_grad = True))
        self.bias = Parameter(init.zeros(self.dim, requires_grad = True))
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        feature_size = x.shape[1]

        mean = x.sum((0, )) / batch_size
        mean_broadcast = mean.broadcast_to(x.shape)
        x_mean_broadcase = x - mean_broadcast
        std = (x_mean_broadcase ** 2).sum((0, )) / batch_size

        if(self.training):
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                                self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                                self.momentum * std
            x_normed = x_mean_broadcase / \
                    ((std + self.eps).broadcast_to(x.shape) ** 0.5) 
        else:
            x_normed = (x - (self.running_mean).broadcast_to(x.shape)) / \
                    ((self.running_var + self.eps).broadcast_to(x.shape) ** 0.5) 
        return x_normed * self.weight.broadcast_to(x.shape) + \
                self.bias.broadcast_to(x.shape) 


class LayerNorm1d(Module):
    """Applies layer normalization over a mini-batch of inputs as described 
    in the paper 'Layer Normalization(https://arxiv.org/abs/1607.06450)'."""
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim # number of channels (the feature size)
        self.eps = eps
        # the learnable weights of size dim, elements initialized to 1.
        self.weight = Parameter(init.ones(self.dim, requires_grad=True))
        # the learnable bias of shape dim, elements initialized to 0 (changed from 1).
        self.bias = Parameter(init.zeros(self.dim, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        """
        x: 2D tensor, batches in the first dimension and features on the second.
        """
        batch_size = x.shape[0]
        feature_size = x.shape[1]  

        # Compute the mean and variance for every batches' feature.
        mean = x.sum(axes = (1, )).reshape((batch_size, 1)) / feature_size
        mean_broadcase = mean.broadcast_to(x.shape)
        x_mean_broadcase = x - mean_broadcase
        std = ((x_mean_broadcase ** 2).sum(axes = (1, ))).reshape((batch_size, 1)) \
                / feature_size + self.eps
        std = std ** 0.5
        std_broadcast = std.broadcast_to(x.shape)

        normed = x_mean_broadcase / std_broadcast

        return self.weight.broadcast_to(x.shape) * normed \
                + self.bias.broadcast_to(x.shape)


class Dropout(Module):
    """During training, randomly zeroes some of the elements of the input tensor
    with probability p using samples from a Bernoulli distribution."""
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # Dropout applies during training only, not during evaluation.
        if self.training:
            mask = init.randb(*x.shape, p = 1-self.p)
            x_mask = x * mask
            return x_mask / (1 - self.p)
        else:
            return x



class Residual(Module):
    """Applies a residual or skip connection given module  F  and input Tensor  x ,
    returning  F(x)+x ."""
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)

