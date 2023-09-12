import operator
import math
from functools import reduce
import numpy as np
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu

# math.prod not in Python 3.7
def prod(x):
    """
    Calculate the multiplications of all elements in list x by mul func 
    in operator lib.
    """
    return reduce(operator.mul, x, 1) # 1 meas the init op.


class BackendDevice:
    """A backend device, wraps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        """Get the name attribute from self class."""
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        # n is count of class, i is index of goal class
        # Get the i row of identity matrix created by numpy eye
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return cpu_numpy()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    """A generic ND array class that may contain multiple different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.
    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.
    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(self, other, device=None):
        """ Create by copying `another NDArray`, or from `numpy` """
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            # ascontiguousarray make other to a contiguous array
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # See if we can create a numpy array from input.
            # Only create a null NDArray.
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        # Create current ndarray five fields from another NDArray.
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle # A memory address point for 1D array in backend.

   
    """ Utils function """
    @staticmethod
    def compact_strides(shape):
        """Utility function to compute compact strides by shape."""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties. This will `allocation the
        memory` if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray) # Instantiate a NDArray class by call __init__.
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None: # Need create a new memory area by <Device API>.
            array._handle = array.device.Array(prod(shape)) # prod(shape) means the size.
        else: # Point a created memory area. 
            array._handle = handle
        return array


    """ Properties of NDArray """
    ### Properies and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        """NOTE: only support float32 for now."""
        return "float32"

    @property
    def ndim(self):
        """ Return number of dimensions. """
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    def __repr__(self):
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()


    """ Basic array manipulation """ 
    def fill(self, value):
        """ <Device API> Fill (in place) with a constant value. """
        self._device.fill(self._handle, value)

    def to(self, device):
        """ Convert between devices, using to/from numpy calls as the unifying bridge. """
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self):
        """ <Device API> convert to a numpy array """
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        """ Return true if array is compact in memory and internal size equals product
        of the shape dimensions """
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self):
        """ <Device API> Convert a matrix to be compact. """
        if self.is_compact():
            return self
        else:
             # The out stride is compacted which is empty.
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape, strides):
        """ Restride the matrix without copying memory. """
        assert len(shape) == len(strides), \
               "[Error NDArray] Require same dimention for shape."
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.
        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.
        Args:
            new_shape (tuple): new shape of the array
        Returns:
            NDArray : reshaped array; this will point to the same memory as the original NDArray.
        """
        assert prod(new_shape) == self.size,  \
               "[Error NDArray] Require same element count of the shape array."
        return self.as_strided(new_shape, NDArray.compact_strides(new_shape))        

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permutation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memory as the original array.
        Args:
            new_axes (tuple): permutation order of the dimensions
        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """
        new_shape   = tuple(self.shape[i]   for i in new_axes)
        new_strides = tuple(self.strides[i] for i in new_axes)
        return self.as_strided(new_shape, new_strides)

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.
        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1
        Args:
            new_shape (tuple): shape to broadcast to
        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """
        assert len(self.shape) == len(new_shape),  \
               "[Error NDArray] Require same dimention for shape."
        new_strides = list(self.strides)
        for i, d in enumerate(self.shape):
            assert d == 1 or new_shape[i] == d,  \
               "[Error NDArray] Require same shape or 1 in every position."
            if d == 1:
                new_strides[i] = 0 
        return self.as_strided(new_shape, tuple(new_strides))


    """ Get and set elements. Slice. """
    def process_slice(self, sl, dim):
        """ 
        Convert a slice to an explicit start/stop/step.
        We only support positive strides and that kind of thing.
        Args:
            sl (slice objetc): a slice object
            dim (int): slice in this dimention
        """
        # Init the iteration variate of the slice.
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0: 
            # If < 0 means iterate in inverted order,
            # so assigns the length of array in that dimension to start.
            start = self.shape[dim]
        if stop == None: # Default assigns the length of array in that dimension
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step) # Return the slice class.

    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array. When passed notation such as a `[1:5,:-1:2,4,:]` etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.
        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory
        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.
        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            such as a `[1:5,:-1:2,4,:]` corresponding to the subset of the matrix 
            to get
        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memory but just
            manipulate the shape/strides/offset of the new array, referencing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple): 
            # For singletons like the '4' in this example convert to tuple
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, \
               "[Error NDArray] Need indexes equal to number of dimensions"

        # Create a new NDArray pointing to the same memory from slices
        new_offset = 0
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        for i, sli in enumerate(idxs):
            new_offset += self.strides[i] * sli.start
            new_shape[i] = math.ceil((sli.stop - sli.start) / sli.step)
            new_strides[i] = self.strides[i] * sli.step
        return NDArray.make( # Slice from same memory address.
            tuple(new_shape), tuple(new_strides), self.device, self._handle, new_offset
        )

    def __setitem__(self, idxs, other):
        """<Device API> Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs) # Get memory of current array.
        if isinstance(other, NDArray): # Set item by NDArray
            assert prod(view.shape) == prod(other.shape), \
                   "[Error NDArray] Size of the array must same."
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else: # Set item by a scalar.
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )


    """ Basic alculation operation by <Device API> """
    ### Collection of elementwise and scalar function: add, multiply, boolean, etc
    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an element-wise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)


    ### Elementwise functions
    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):
        """Matrix multiplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.
        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will re-stride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size.
        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2,  \
               "[Error NDArray] Only support 2D array."
        assert self.shape[1] == other.shape[0],     \
               "[Error NDArray] Shape constrans error."

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # If the matrix is aligned, use tiled matrix multiplication.
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    # shape was tiled
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile), 
                    # Stride, but don't change the memory address.
                    (a.shape[1] * tile, tile, self.shape[1], 1), 
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)

            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3)).compact() # Get correct row-order of matirx. 
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions utils function, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis, keepdims=False):
        """ 
        Return a view to the array set up for reduction functions and output array. 
        Which simplize the function work by permute array, that make the axis become
        last dimension.
        """
        if isinstance(axis, tuple) and not axis:
            axis = None

        if axis is None:
            # view is the self converted to (1, 1, ..., prod(self.shape)), which 
            # indicates the multimensional shape but only last dimension have 
            # all elements.
            view = self.reshape((1,) * (self.ndim - 1) + (prod(self.shape),)) 
            # Create a null array out which dimentions is same to view.
            out = NDArray.make((1,) * (self.ndim if keepdims else 1), device=self.device) 
        else:
            # Convert the dimension of axis to be the last dimension by permute array. 
            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make( # Only sum axis dimension become 1 for out array.
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),
                device=self.device,
            ) 
        return view, out

    def sum(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out


""" Global function. Handle the NDArray as args. """
def array(a, dtype="float32", device=None):
    """ Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32", "[Error NDArray] Only support FP32."
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array: NDArray, new_shape):
    return array.broadcast_to(new_shape)


def reshape(array: NDArray, new_shape):
    return array.reshape(new_shape)


def maximum(a: NDArray, b: NDArray):
    return a.maximum(b)


def log(a: NDArray):
    return a.log()


def exp(a: NDArray):
    return a.exp()


def tanh(a: NDArray):
    return a.tanh()


# def flip(a, axes):
#     return a.flip(axes)


def summation(a: NDArray, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)


def matmul(a: NDArray, b: NDArray):
    return a @ b


def max(a: NDArray, axis=None, keepdims=False):
    return a.max(axis=axis, keepdims=keepdims)