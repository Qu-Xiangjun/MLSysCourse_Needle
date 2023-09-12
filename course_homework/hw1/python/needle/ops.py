"""Operator implementations."""
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    # use compute func in make_from_op called by TensorOp __call__.
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
        if(self.axes):
            return array_api.swapaxes(a, *self.axes)
        return array_api.swapaxes(a, -1, -2) # Default to swap axis -1 and -2 dimention.

    def gradient(self, out_grad, node):
        if self.axes: 
            return transpose(out_grad, self.axes) # Return new tensor for trans grad.
        else: 
            return transpose(out_grad)

def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    """Give a new shape to an array without changing its data."""
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        shape = node.inputs[0].shape
        return reshape(out_grad, shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    """Broadcast an array to a new shape."""
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Get the input tensor shape
        shape = list(node.inputs[0].shape)  
        # The inputs shape will be aligned with the lower part of the node.shape. 
        # Construct a shape list, which len same to node.shape.
        # Inputs shape's len will less than or equal to node.shape,
        # so add o to high position of the constructed shape.
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
    """Sum of array elements over given axes (1 input, axes - tuple)."""
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Find which axes is reduced in sum operator, 
        # and broadcast they by inputs shape.
        # Fristly reshape the out_grad to the input shape dimention,
        # so that it can directly broadcast, not shape error.
        # Such as shape (5, 4) and axis = (1,) sumed to (5,), which firstly 
        # reshape to (5, 1) and then broadcast to (5, 4).
        in_shape = node.inputs[0].shape
        
        # Get which the axes set.
        if self.axes:
            s = set(self.axes)
        else:
            # All the axes sumed.
            s = set(range(len(in_shape)))

        # Construct the re_shape, which init to list full of 1, and keep same
        # shape value with the out_grad shape value in the index which is not in axes.
        re_shape = [1] * len(in_shape)
        j = 0
        for i in range(len(in_shape)):
            if i not in s:
                re_shape[i] = out_grad.shape[j]
                j += 1
        return broadcast_to(reshape(out_grad, tuple(re_shape)), in_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    """Matrix multiplication of the inputs."""
    def compute(self, a, b):
        # Notice the matirx @ vector will be vector,
        # and the vector will transpose defaultly. 
        return a @ b

    def gradient(self, out_grad, node):
        # lhs:(M, K) @ rhs:(K, N) -> node:(M, N)
        lhs, rhs = node.inputs[0], node.inputs[1]
        # # TODO: The matirx @ vector don't contain in test case.
        # # Notice there is not contain situation for matirx @ vector -> vector.
        # vector_flag = False
        # if(len(rhs.shape) == 1):
        #     vector_flag = True
        #     # print("**************************True*********************")
        #     print("a.shape",lhs.shape)
        #     print("b.shape",rhs.shape)
        #     print("node.shape",node.shape)
        #     grad_a = matmul(reshape(out_grad, (out_grad.shape[0], 1)),
        #          reshape(rhs, (1, rhs.shape[0])))
        #     grad_b = matmul(transpose(lhs), out_grad)
        #     print("grad_a.shape",grad_a.shape)
        #     print("grad_b.shape",grad_b.shape)

        # Notice the transpose func default dimention -1 and -2.
        # grad_a:(M, K) == node:(M, N) @ rhs:(K, N).T
        grad_a = matmul(out_grad, transpose(rhs))
        # grad_b:(K, N)== rhs:(M, K).T @ node:(M, N)
        grad_b = matmul(transpose(lhs), out_grad)
    
        # Notice the shape is changed with inputs shape in batch.
        # Such as (6, 6, 5, 4) @ (4, 3) -> (6, 6, 5, 3), and the 
        # grad_b = (6, 6, 4, 3) must be sumed.
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
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * array_api.exp(node.inputs[0].realize_cached_data())


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return out_grad * (node.inputs[0].realize_cached_data() > 0).astype("float32")


def relu(a):
    return ReLU()(a)

