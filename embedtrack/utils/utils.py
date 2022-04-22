"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
from tifffile import imread, imsave


def collect_paths(path_to_dir):
    """Returns a list of full paths to the lowest subdirectories of the provided path"""
    folder_content = os.walk(path_to_dir)
    sub_paths = [sub_path[0] for sub_path in folder_content if not sub_path[1]]
    for index in range(len(sub_paths)):
        sub_paths[index] = sub_paths[index].replace("\\", "/")
    return sub_paths


def compress_tifffiles(data_path):
    """
    Compresses tiff files in a folder.
    Args:
        data_path: path to a folder containing tiff files

    """
    data_path = Path(data_path)
    for f in os.listdir(data_path):
        if f.endswith(".tif") or f.endswith(".tiff"):
            imsave(data_path / f, imread(data_path / f), compress=2)


def decompress_tifffiles(data_path):
    """
    Decompresses tiff files in a folder.
    Args:
        data_path: path to a folder containing tiff files

    """
    data_path = Path(data_path)
    for element in os.listdir(data_path):
        if element.endswith(".tif") or element.endswith(".tiff"):
            imsave(data_path / element, imread(data_path / element))


def collect_leaf_paths(root_paths):
    """Collects all paths to leaf folders."""
    leaf_paths = [
        p for p in Path(root_paths).glob("**") if not os.walk(p).__next__()[1]
    ]
    return leaf_paths


def get_indices_pandas(data, background_id=0):
    """
    Extracts for each mask id its positions within the array.
    Args:
        data: a np. array with masks, where all pixels belonging to the
            same masks have the same integer value
        background_id: integer value of the background

    Returns: data frame: indices are the mask id , values the positions of the mask pixels

    """
    if data.size < 1e9:  # aim for speed at cost of high memory consumption
        masked_data = data != background_id
        flat_data = data[masked_data]  # d: data , mask attribute
        dummy_index = np.where(masked_data.ravel())[0]
        df = pd.DataFrame.from_dict({"mask_id": flat_data, "flat_index": dummy_index})
        df = df.groupby("mask_id").apply(
            lambda x: np.unravel_index(x.flat_index, data.shape)
        )
    else:  # aim for lower memory consumption at cost of speed
        flat_data = data[(data != background_id)]  # d: data , mask attribute
        dummy_index = np.where((data != background_id).ravel())[0]
        mask_indices = np.unique(flat_data)
        df = {"mask_id": [], "index": []}
        data_shape = data.shape
        for mask_id in mask_indices:
            df["index"].append(
                np.unravel_index(dummy_index[flat_data == mask_id], data_shape)
            )
            df["mask_id"].append(mask_id)
        df = pd.DataFrame.from_dict(df)
        df = df.set_index("mask_id")
        df = df["index"].apply(lambda x: x)  # convert to same format as for other case
    return df


def get_img_files(img_dir, starts_with=""):
    """
    Extracts a set of tiff files from a folder.
    Args:
        img_dir: path to the image folder
        starts_with: optional string the image name needs to start with

    Returns:

    """
    img_file_pattern = re.compile(
        r"(\D*)(\d+)(\D*)\.(" + "|".join(("tif", "tiff")) + ")"
    )
    files = {
        int(img_file_pattern.match(file).groups()[1]): (img_dir / file).as_posix()
        for file in os.listdir(img_dir)
        if file.endswith(("tif", "tiff")) and file.startswith(starts_with)
    }
    return files
