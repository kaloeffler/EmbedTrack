"""
Original work Copyright 2019 Davy Neven,  KU Leuven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))
Modified work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
Modifications: processing of image pairs; augmentation of offset maps; Blur,Clahe, min max percentile augmentation
"""
import collections

import albumentations
import numpy as np
import torch
from torchvision.transforms import transforms as T
from skimage.exposure import equalize_adapthist
from skimage import img_as_float32


class RandomRotationsAndFlips(T.RandomRotation):
    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys

    def __call__(self, sample):

        angle = self.get_params(self.degrees)
        times = np.random.choice(4)
        flip = np.random.choice(2)
        for idx, k in enumerate(self.keys):
            assert k in sample
            k_name = k.split("_")[0]

            # same rot for flow image cxhxw
            if k_name == "flow":
                temp = np.ascontiguousarray(np.rot90(sample[k], times, (1, 2)))
                if times == 2:
                    temp *= -1
                if (
                    times % 2
                ):  # 270,90 rot -> row will be col and col will be row -> flip channel
                    temp = -temp[::-1, ...].copy()
                    # 90 deg
                    if times == 1:
                        temp[1, ...] *= -1
                    if times == 3:
                        temp[0, ...] *= -1

            else:
                temp = np.ascontiguousarray(np.rot90(sample[k], times, (1, 2)))
            if flip == 0:
                sample[k] = temp
            else:

                if k_name == "flow":
                    temp = np.ascontiguousarray(
                        np.flip(temp, axis=1)
                    )  # flip about D - axis, change flow dir of x
                    temp[0, ...] *= -1
                    sample[k] = temp
                else:
                    sample[k] = np.ascontiguousarray(
                        np.flip(temp, axis=1)
                    )  # flip about Y - axis
        return sample


class ImageAugmentation:
    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys
        self.transform = [
            albumentations.Blur(p=0.3),
            Clahe(p=0.3),
        ]

    def __call__(self, sample):
        for idx, k in enumerate(self.keys):
            assert k in sample
            if k.startswith("image"):
                sample[k] = self.transform_image(sample[k])

        return sample

    def transform_image(self, image):
        temp = image.copy()
        img_shape = temp.shape
        temp = img_as_float32(temp)

        for transform in self.transform:
            temp = (
                transform(image=temp.squeeze())["image"]
                .reshape(img_shape)
                .astype(np.float32)
            )

        return temp.reshape(img_shape)


class Clahe:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if np.random.rand() <= self.p:
            return {"image": equalize_adapthist(image)}
        return {"image": image}


# from github.com/Mouseland/cellpose/omnipose/utils.py
def normalize(img, lower=0.01, upper=99.99):
    lower_perc = np.percentile(img, lower)
    upper_perc = np.percentile(img, upper)
    return np.interp(img, (lower_perc, upper_perc), (0, 1))


class ToTensorFromNumpy(object):
    def __init__(self, keys=[], type="float"):

        if isinstance(type, collections.Iterable):
            assert len(keys) == len(type)

        self.keys = keys
        self.type = type

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):
            t = self.type
            if isinstance(t, collections.Iterable):
                t = t[idx]
            if k in sample:
                k_name = k.split("_")[0]
                if k_name == "image":  # image
                    sample[k] = torch.from_numpy(
                        normalize(sample[k].astype("float32"), lower=1.0, upper=99.0)
                    ).float()
                if k_name == "flow":
                    sample[k] = torch.from_numpy(sample[k].astype("float32")).float()
                elif (
                    k_name == "instance"
                    or k_name == "label"
                    or k_name == "center-image"
                ):
                    sample[k] = torch.from_numpy(sample[k]).short()

        return sample


def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr["name"]
        opts = tr["opts"]

        transform_list.append(globals()[name](**opts))

    return T.Compose(transform_list)
