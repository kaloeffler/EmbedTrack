"""
Original work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
Modifications: process image pairs; shift augmentation; process multiple data sets
"""
import glob
import os
import random
from pathlib import Path

import numpy as np
import tifffile
from torch.utils.data import Dataset


class TwoDimensionalDataset(Dataset):
    """
    TwoDimensionalDataset class
    """

    def __init__(
        self,
        crop_dir="./",
        data_dir="./",
        data_subsets=[],
        center="center-medoid",
        type="train",
        bg_id=0,
        size=None,
        transform=None,
        translation_prob=0.5,
        max_rel_translation=0.1,
    ):
        """

        Args:
            crop_dir (string): path to the directory containing the cropped data
            data_dir (string): path to the directory containing the non cropped data
            data_subsets (list): list of sub_directories to use
            center (string): indicating the type of cell centers to train on
            type (string): data set type ("train" or "val")
            bg_id (int): value of background pixels in the segmentation masks
            size (int, optional):  if int yield only N samples of the dataset per (train/val) epoch;
                                    if None: use the full dataset  per (train/val) epoch
            transform (Callable): transformations to apply to each sample
            translation_prob (float): probability of a sample to be translated in a random direction
            max_rel_translation (float): maximum translation relative to the crop size (e.g. if max_rel_translation=0.1
             and the crop size is 256 images can be shifted by a maximum of 25 pixels to each other)
        """

        print(
            "2-D `{}` dataloader created! Accessing data from {}/{}/".format(
                type, crop_dir, data_subsets
            )
        )
        self.data_dir = data_dir
        self.center = center
        # get image and instance list
        image_list = []
        instance_list = []
        center_image_list = []
        flow_image_list = []
        for sub_set in data_subsets:
            img_list = glob.glob(
                os.path.join(crop_dir, "{}/".format(sub_set), "images/*.tif")
            )
            img_list.sort()
            image_list.extend(img_list)
            print(
                "Number of images in `{}` directory is {}".format(
                    sub_set, len(img_list)
                )
            )

            inst_list = glob.glob(
                os.path.join(crop_dir, "{}/".format(sub_set), "masks/*.tif")
            )
            print(
                "Number of instances in `{}` directory is {}".format(
                    sub_set, len(inst_list)
                )
            )
            inst_list.sort()
            instance_list.extend(inst_list)

            center_img_list = glob.glob(
                os.path.join(crop_dir, "{}/".format(sub_set), center + "/*.tif")
            )
            print(
                "Number of center images in `{}` directory is {}".format(
                    sub_set, len(center_img_list)
                )
            )
            center_img_list.sort()
            center_image_list.extend(center_img_list)

            flow_img_list = glob.glob(
                os.path.join(
                    crop_dir, "{}/".format(sub_set), center + "-flow" + "/*.tif"
                )
            )
            flow_image_list.extend(flow_img_list)

        print("*************************")

        self.image_list = image_list
        self.instance_list = instance_list
        self.center_image_list = center_image_list
        self.flow_image_list = flow_image_list
        self.bg_id = bg_id
        self.size = size
        self.pair_index = self.get_image_pairs()
        self.n_pairs = len(self.pair_index)
        self.transform = transform
        self.crop_size = None
        self.p_translation = translation_prob
        self.max_offset = max_rel_translation

    def get_image_pairs(self):
        """
        Extract all pairs of image crops of successive time points t, t-1
        Returns:

        """
        # same name for all crops: masks, images, center_images
        pairs = []
        for i, path_img_file in enumerate(self.image_list):
            path_img, name_img = os.path.split(path_img_file)
            name_img, ending = name_img.split(".")
            time, patch_id = name_img.split("_")
            name_next_img = (
                "_".join([str(int(time) + 1).zfill(len(time)), patch_id]) + "." + ending
            )
            flow_img_name = (
                "_".join([str(int(time) + 1).zfill(len(time)), time, patch_id])
                + "."
                + ending
            )
            path_flow_img = os.path.join(
                os.path.dirname(path_img),
                "-".join([self.center, "flow"]),
                flow_img_name,
            )

            try:
                pairs.append(
                    (
                        self.image_list.index(os.path.join(path_img, name_next_img)),
                        i,
                        self.flow_image_list.index(path_flow_img),
                    )
                )
            except ValueError:
                continue
        return pairs

    def __len__(self):

        return len(self.pair_index) if self.size is None else self.size

    def convert_yx_to_cyx(self, im, key):
        if im.ndim == 2 and key == "image":  # gray-scale image
            im = im[np.newaxis, ...]  # CYX
        elif im.ndim == 3 and key == "image":  # multi-channel image image
            pass
        else:
            im = im[np.newaxis, ...]
        return im

    def get_sample(self, index):
        """
        Get a data sample - containing the raw image pair (t, t-1) and the label images (t, t-1)
        Args:
            index (int): index of the selected image pair

        Returns: dict of data sample

        """
        index_curr, index_prev, flow_index = self.pair_index[index]
        sample = {}

        # load image pair
        image_curr = tifffile.imread(self.image_list[index_curr])  # YX or CYX
        image_prev = tifffile.imread(self.image_list[index_prev])  # YX or CYX
        image_curr = self.convert_yx_to_cyx(image_curr, key="image")
        image_prev = self.convert_yx_to_cyx(image_prev, key="image")
        sample["image_curr"] = image_curr  # CYX
        sample["image_prev"] = image_prev  # CYX
        sample["im_name_curr"] = self.image_list[index_curr]
        sample["im_name_prev"] = self.image_list[index_prev]
        if len(self.instance_list) != 0:
            instance_curr = tifffile.imread(
                self.instance_list[index_curr]
            )  # YX or DYX (one-hot!)
            instance_prev = tifffile.imread(
                self.instance_list[index_prev]
            )  # YX or DYX (one-hot!)
            instance_curr, label_curr = self.decode_instance(instance_curr, self.bg_id)
            instance_prev, label_prev = self.decode_instance(instance_prev, self.bg_id)
            instance_curr = self.convert_yx_to_cyx(
                instance_curr, key="instance"
            )  # CYX or CDYX
            instance_prev = self.convert_yx_to_cyx(
                instance_prev, key="instance"
            )  # CYX or CDYX
            label_curr = self.convert_yx_to_cyx(label_curr, key="label")  # CYX
            label_prev = self.convert_yx_to_cyx(label_prev, key="label")  # CYX
            sample["instance_curr"] = instance_curr
            sample["instance_prev"] = instance_prev
            sample["label_curr"] = label_curr
            sample["label_prev"] = label_prev
        if len(self.center_image_list) != 0:
            center_image_curr = tifffile.imread(
                self.center_image_list[index_curr]
            ).astype(np.bool)
            center_image_prev = tifffile.imread(
                self.center_image_list[index_prev]
            ).astype(np.bool)
            center_image_curr = self.convert_yx_to_cyx(
                center_image_curr, key="center_image"
            )  # CYX
            center_image_prev = self.convert_yx_to_cyx(
                center_image_prev, key="center_image"
            )  # CYX
            sample["center_image_curr"] = center_image_curr
            sample["center_image_prev"] = center_image_prev
            sample["flow"] = tifffile.imread(self.flow_image_list[flow_index])
        return sample

    def get_translated_sample(self, index):
        """
        Simulate a shift between successive frames by shifting an image crop in a random direction.
        Args:
            index (int): index of the selected image pair, however only the first image will be used and translated

        Returns:

        """
        # get first image
        img_id_curr, _, _ = self.pair_index[index]
        img_file = self.image_list[img_id_curr]
        if self.crop_size is None:
            self.crop_size = tifffile.imread(img_file).shape
        # sample position
        img_crop_name = os.path.basename(img_file)
        full_img_name = img_crop_name.split("_")[0] + "." + img_crop_name.split(".")[-1]
        data_set = Path(self.data_dir).name
        path_parts = list(Path(img_file).parts)
        sub_dirs = path_parts[
            path_parts.index(data_set) + 1 : path_parts.index("images")
        ]
        image = tifffile.imread(
            os.path.join(self.data_dir, *sub_dirs, "images", full_img_name)
        )
        center_image = tifffile.imread(
            os.path.join(self.data_dir, *sub_dirs, self.center, full_img_name)
        )
        instance_image = tifffile.imread(
            os.path.join(self.data_dir, *sub_dirs, "masks", full_img_name)
        )
        # calculate a shift and crop the image to the crop size
        height, width = image.shape
        if len(self.crop_size) != 2:
            raise AssertionError(f"{self.crop_size} is not 2D")
        top_left = (
            np.random.randint(height - self.crop_size[0]),
            np.random.randint(width - self.crop_size[1]),
        )
        # calculation of the min max bounds of the offset
        translation = (
            np.random.randint(
                -min(top_left[0], self.max_offset * self.crop_size[0]),
                min(
                    height - self.crop_size[0] - top_left[0] - 1,
                    self.max_offset * self.crop_size[0],
                ),
            ),
            np.random.randint(
                -min(top_left[1], self.max_offset * self.crop_size[1]),
                min(
                    width - self.crop_size[1] - top_left[1],
                    self.max_offset * self.crop_size[1],
                ),
            ),
        )
        top_left_translated = (
            top_left[0] + translation[0],
            top_left[1] + translation[1],
        )
        image_curr = self.get_image_crop(image, top_left_translated)
        center_image_curr = self.get_image_crop(center_image, top_left_translated)
        instance_curr = self.get_image_crop(instance_image, top_left_translated)

        image_prev = self.get_image_crop(image, top_left)
        center_image_prev = self.get_image_crop(center_image, top_left)
        instance_prev = self.get_image_crop(instance_image, top_left)

        offset = np.zeros((len(self.crop_size), *self.crop_size))
        offset[:, center_image_curr != 0] = -np.array(translation).reshape(-1, 1)

        sample = dict()
        sample["image_curr"] = self.convert_yx_to_cyx(image_curr, key="image")  # CYX
        sample["image_prev"] = self.convert_yx_to_cyx(image_prev, key="image")  # CYX
        sample["im_name_curr"] = full_img_name
        sample["im_name_prev"] = full_img_name
        if len(self.instance_list) != 0:
            instance_curr, label_curr = self.decode_instance(instance_curr, self.bg_id)
            instance_prev, label_prev = self.decode_instance(instance_prev, self.bg_id)
            instance_curr = self.convert_yx_to_cyx(
                instance_curr, key="instance"
            )  # CYX or CDYX
            instance_prev = self.convert_yx_to_cyx(
                instance_prev, key="instance"
            )  # CYX or CDYX
            label_curr = self.convert_yx_to_cyx(label_curr, key="label")  # CYX
            label_prev = self.convert_yx_to_cyx(label_prev, key="label")  # CYX
            sample["instance_curr"] = instance_curr
            sample["instance_prev"] = instance_prev
            sample["label_curr"] = label_curr
            sample["label_prev"] = label_prev
        if len(self.center_image_list) != 0:
            center_image_curr = self.convert_yx_to_cyx(
                center_image_curr, key="center_image"
            )  # CYX
            center_image_prev = self.convert_yx_to_cyx(
                center_image_prev, key="center_image"
            )  # CYX
            sample["center_image_curr"] = center_image_curr
            sample["center_image_prev"] = center_image_prev
            sample["flow"] = offset
        return sample

    def get_image_crop(self, image, top_left):
        return image[
            top_left[0] : top_left[0] + self.crop_size[0],
            top_left[1] : top_left[1] + self.crop_size[1],
        ]

    def __getitem__(self, index):
        if self.n_pairs == 1:
            index = 0
        else:
            index = (
                index
                if self.size is None
                else random.randint(0, len(self.pair_index) - 1)
            )
        if np.random.rand() <= self.p_translation:
            sample = self.get_translated_sample(index)
        else:
            sample = self.get_sample(index)
            # transform
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    @classmethod
    def decode_instance(cls, pic, bg_id=None):
        pic = np.array(pic, copy=False, dtype=np.uint16)
        instance_map = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.int16)
        class_map = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if bg_id is not None:
            mask = pic > bg_id
            if mask.sum() > 0:
                instance_map[mask] = pic[mask]
                class_map[mask] = 1

        return instance_map, class_map


def get_dataset(name, dataset_opts):
    if name == "2d":
        return TwoDimensionalDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))
