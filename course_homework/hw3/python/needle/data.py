import numpy as np
from .autograd import Tensor
import gzip
import needle as ndl

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return img[:, :, ::-1]
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        
        result = ndl.zeros_like(img)
        H, W = img.shape[0], img.shape[1]

        # Shift the all img from the origin canvas.
        if (abs(shift_x) >= H or abs(shift_y) >= W):
            return result

        # Shift a part of img from the origin canvas.
        # Rnage in canvans
        down, up = max(0, -shift_x), min(H - shift_x, H)
        left, right = max(0, -shift_y), min(W - shift_y, W)
        # Rnage in img
        img_down, img_up = max(0, shift_x), min(H + shift_x, H)
        img_left, img_right = max(0, shift_y), min(W + shift_y, W)
        result[down:up, left:right, :] = img[img_down:img_up, img_left:img_right, :]
        return result


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )
        else:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), 
                                           range(self.batch_size, len(self.dataset), 
                                           self.batch_size))


    # The first get a iterator object and then call `__next__` until `StopIteration`.
    def __iter__(self):
        self.start = 0
        return self


    def __next__(self):
        if(self.start == len(self.ordering)):
            raise StopIteration
        bacth_index = self.start
        self.start += 1
        samples = [Tensor(x) for x in self.dataset[self.ordering[bacth_index]]]
        return tuple(samples)


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        self.image, self.labels = parse_mnist(image_filename, label_filename)

    def __getitem__(self, index) -> object:
        X, y = self.image[index], self.labels[index]
        if self.transforms:
            X_in = X.reshape((28, 28, -1))
            X_out = self.apply_transforms(X_in)
            X_ret = X_out.reshape(-1, 28 * 28)
            return X_ret, y
        else:
            return X, y

    def __len__(self) -> int:
        return self.labels.shape[0]


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


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
