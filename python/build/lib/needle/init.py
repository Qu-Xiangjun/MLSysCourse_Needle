import math
import needle as ndl


def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random numbers uniform between low and high """
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape, dtype=dtype) * (high - low) + low
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    device = ndl.cpu() if device is None else device
    array = device.randn(*shape, dtype=dtype) * std + mean
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate constant Tensor """
    device = ndl.cpu() if device is None else device
    array = device.full(shape, c, dtype=dtype)
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape) <= p
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """ Generate one-hot encoding Tensor """
    device = ndl.cpu() if device is None else device
    return ndl.Tensor(
        device.one_hot(n, i.numpy(), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    """
    Fills the input Tensor with values using a uniform(-a, a) distribution.
    uniform(-a, a), a = gain * pow(6 /( fan_in + fan_out), 2))
    Args:
        fan_in(Int): dimensionality of input
        fan_out(Int): dimensiionality of output
        gain(Float): option scaling factor
    Return: 
        Tensor: the input Tensor  
    """
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return 2 * a * rand(fan_in, fan_out, **kwargs).data - a;


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    """
    Fills the input Tensor with values using a normal(0, std^2) distribution.
    normal(0, std^2), std = gain * sqrt(2 /( fan_in + fan_out)))
    Args:
        fan_in(Int): dimensionality of input
        fan_out(Int): dimensiionality of output
        gain(Float): option scaling factor
    Return: 
        Tensor: the input Tensor  
    """
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean = 0.0, std = std, **kwargs)


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = math.sqrt(6 / fan_in)
    return 2 * bound * rand(fan_in, fan_out, **kwargs).data - bound;


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    std = math.sqrt(2 / fan_in)
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)

