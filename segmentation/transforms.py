import random

from torchvision.transforms import functional
import torchvision.transforms as T 

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = functional.resize(image, self.size)
        target = functional.resize(target, self.size)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = functional.resize(image, size)
        target = functional.resize(target, size, interpolation=functional.InterpolationMode.NEAREST)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = functional.crop(image, *crop_params)
        target = functional.crop(target, *crop_params)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = functional.hflip(image)
            target = functional.hflip(target)
        return image, target

class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = functional.vflip(image)
            target = functional.vflip(target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = functional.to_tensor(image)
        target = functional.to_tensor(target)
        return image, target

