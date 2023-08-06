"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        return (out_grad * self.scalar * (lhs ** (self.scalar - 1)))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs[0], node.inputs[1]
        grad_a = out_grad / rhs
        grad_b = out_grad * lhs * (-1) / (rhs ** 2)
        return (grad_a, grad_b)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    """Op of the input by a scalar, element-wise (1 input, scalar - number)."""
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar, )


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    """reverses the order of two axes (axis1, axis2), 
    defaults to the last two axes (1 input, axes - tuple)."""
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        order = list(range(len(a.shape)))
        if(self.axes):
            order[self.axes[0]] = self.axes[1]
            order[self.axes[1]] = self.axes[0]
        else:
            order[-1], order[-2] = order[-2], order[-1]
        return a.permute(tuple(order)) # Default to swap axis -1 and -2 dimention.

    def gradient(self, out_grad, node):
        if self.axes: 
            return transpose(out_grad, self.axes) # Return new tensor for trans grad.
        else: 
            return transpose(out_grad)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        shape = node.inputs[0].shape
        return reshape(out_grad, shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        if len(len(a.shape) != len(self.shape)):
            # Construct a new shape, which lenth same to goal shape.
            # And value of new shape is 1 or same elements between two shape.
            # Notice 1 only add to high position.
            new_shape = [1] * (len(self.shape) - len(a.shape)) + a.shape
            a_ = reshape(a, new_shape)
        else:
            a_ = a
        
        return a_.broadcast_to(self.shape)

    def gradient(self, out_grad, node):
        # Get the input tensor shape
        shape = list(node.inputs[0].shape)  
        # The inputs shape will be aligned with the lower part of the node.shape. 
        # Construct a shape list, which len same to node.shape.
        # Inputs shape's len will less than or equal to node.shape,
        # so add 1 to high position of the constructed shape.
        # Then, every index for constructed shape differented from node.shape 
        # will be sum in out_grad, which is the inputs tensor grad.
        # For example,inputs shape (2, 1) brodcast to node shape(2, 2, 3), 
        # the constucted shape will be (1, 2, 1), so the axes(0, 2) will be sumed.
        shape = [1] * (len(node.shape) - len(shape)) + shape
        axes = [] # Store the axes which is need to broadcast.
        for i, s in enumerate(node.shape):
            # Find the axes needed to broadcast.The shape value is different from 
            # node.shape value in that axes.
            if i >= len(shape) or s != shape[i]:
                axes.append(i)
        # The output gradient tensor is summed along the axis to be broadcast.
        # Notice using needle.tensor api to consturct a new tensor, not change out_grad.
        summed = summation(out_grad, tuple(axes))
        # Adjust the sum result to the shape of the input tensor by needle.tensor api.
        grad = reshape(summed, node.inputs[0].shape)
        return grad


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        if self.axes is None:
            return a.sum()
        else:
            # NOTE self.axes maybe int
            if isinstance(self.axes, int):
                return a.sum(self.axes) 
            # NOTE only support sum in a single dim
            for i, axis in enumerate(sorted(list(self.axes))):
                # NOTE -i because each sum() operation will reduce the dimension number
                a = a.sum(axis-i)
            return a

    def gradient(self, out_grad, node):
        shape = node.inputs[0].shape

        # Get the out_shape to broadcast
        if self.axes is None:
            shape_out = [1 for i in range(shape)]
        else:
            if isinstance(self.axes, int):
                axes = [self.axes]
            else:
                axes = self.axes
            shape_out = shape
            for index in axes:
                shape_out[index] = 1 # have been sumed.
        
        return broadcast_to(reshape(out_grad, tuple(shape_out)), shape)



def summation(a, axes=None):
    return Summation(axes)(a)


class Max(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims = False):
        self.axes = axes
        self.keepdims = keepdims
    
    def compute(self, a: NDArray):
        return a.max(axis = self.axes, keepdims = self.keepdims)
    
    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0].realize_cached_data()
        output_tensor = input_tensor.max(axis = self.axes, keepdims = True)
        output_tensor = Tensor(output_tensor, device=out_grad.device, dtype=out_grad.dtype)
        grad_input = out_grad * (input_tensor == output_tensor)
        return grad_input


def max(a, axes=None):
    return Max(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        grad_a = matmul(out_grad, transpose(rhs))
        grad_b = matmul(transpose(lhs), out_grad)
        if grad_a.shape != lhs.shape: 
            length = len(grad_a.shape) - len(lhs.shape)
            grad_a = summation(grad_a, axes=tuple(range(length)))
        if grad_b.shape != rhs.shape:
            length = len(grad_b.shape) - len(rhs.shape)
            grad_b = summation(grad_b, axes=tuple(range(length)))
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return array_api.exp(node.inputs[0].cached_data) * out_grad


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a: NDArray):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        mask = Tensor(a > 0, device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
        return out_grad * mask  

def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z: NDArray):
        Z_max = array_api.max(Z, axis = self.axes, keepdims=False)
        Z_max_broadcast = array_api.max(Z, axis = self.axes, keepdims=False)
        ret = Z_max + array_api.log(
            array_api.summation(array_api.exp(Z - Z_max_broadcast), axis = self.axes)
        )

    def gradient(self, out_grad, node):
        Z = node.inputs[0].realize_cached_data()
        Z_max_broadcast = array_api.max(Z, axis = self.axes, keepdims = True)
        exp_Z_maxZ = array_api.exp(Z - Z_max_broadcast)
        sum_exp_Z_maxZ = array_api.summation(exp_Z_maxZ, axis = self.axes)

        # First compute the grad for out * grad for log operator, 
        # because their dimension is same.
        log_grad = out_grad.realize_cached_data() / sum_exp_Z_maxZ
        # Make the log_grad's shape reshape to the input shape by add dimension.
        shape = [1] * len(Z.shape)
        if self.axes:
            s = set(self.axes)
            j = 0
            for i in range(len(shape)):
                if j not in s:
                    shape[i] = node.shape[j]
                    j += 1
        log_grad_reshape = log_grad.reshape(tuple(shape))
        return Tensor(log_grad_reshape * exp_Z_maxZ)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a: NDArray):
        return a.tanh()

    def gradient(self, out_grad, node):
        return (1 - tanh(node.inputs[0])**2) * out_grad


def tanh(a):
    return Tanh()(a)