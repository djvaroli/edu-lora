import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from .transforms import flatten_transform, scale_tanh_range


def load_flat_mnist_datasets() -> tuple[MNIST, MNIST]:
    transforms = Compose([ToTensor(), flatten_transform(), scale_tanh_range()])

    mnist_train = MNIST(root=".", transform=transforms, download=True)

    mnist_eval = MNIST(root=".", train=False, transform=transforms)

    return mnist_train, mnist_eval
