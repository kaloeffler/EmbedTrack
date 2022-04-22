"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread, imsave

from embedtrack.utils.utils import (
    collect_leaf_paths,
    collect_paths,
    get_indices_pandas,
)
import os


def copy_and_rename_data(
    raw_data_dir, result_dir, keep_st=True, sub_dir_names=["01", "02"]
):
    """
    Copy ctc (cell tracking challenge) dataset and keep either ST or GT annotation masks.
    Args:
        raw_data_dir (Path): path where the original CTC dataset is stored
        result_dir (Path): path where a modified copy of the dataset will be stored
        keep_st (bool): use ST (silver truth) or GT (gold truth) annotation
        sub_dir_names (list): list of image sequences to copy
    """
    result_dir = result_dir
    sub_directories = [
        element
        for element in os.listdir(raw_data_dir)
        if os.path.isdir(raw_data_dir / element)
    ]
    # copy relevant data
    for sub_dir in sub_directories:
        copy_sub_dir = (
            sub_dir.isnumeric()
            or sub_dir == (sub_dir.split("_")[0] + "_ST")
            or sub_dir == (sub_dir.split("_")[0] + "_GT")
        )
        if copy_sub_dir:
            shutil.copytree(raw_data_dir / sub_dir, result_dir / sub_dir)
    if keep_st:
        rm_dir = "_GT"
    else:
        rm_dir = "_ST"

    keep_dirs = [sub_dir for sub_dir in os.listdir(result_dir) if rm_dir not in sub_dir]
    img_dirs = [sub_d for sub_d in keep_dirs if sub_d.isnumeric()]
    mask_dirs = [sub_d for sub_d in keep_dirs if not sub_d.isnumeric()]
    # sort sub dirs to avoid shutil.move issues e.g. like moving existing directories
    keep_dirs = img_dirs + mask_dirs

    for sub_dir_name in sub_dir_names:
        if not os.path.exists(result_dir / sub_dir_name):
            os.mkdir(result_dir / sub_dir_name)

    # save masks (0x_ST or =x_GT) to mask folder and raw images (0x folder) to images
    # fuse the ST segmentation annotations with the gt lineage annotations
    for sub_dir in keep_dirs:
        sub_dir_name = sub_dir.split("_")[0]
        assert sub_dir_name.isnumeric(), f"unknown directory naming {sub_dir}"
        if sub_dir.isnumeric():
            dir_name = "images"
        else:
            dir_name = "masks"
            if keep_st:
                gt_dir = sub_dir.split("_")[0] + "_GT"
                generate_st_gt_lineage(
                    result_dir / sub_dir / "SEG", result_dir / gt_dir / "TRA"
                )
            else:
                sub_sub_dir = [
                    element
                    for element in os.listdir(result_dir / sub_dir)
                    if os.path.isdir(result_dir / sub_dir / element)
                ]
                for d in sub_sub_dir:
                    if d != "TRA":
                        shutil.rmtree(result_dir / sub_dir / d)
        data_path = result_dir / sub_dir_name / dir_name
        try:
            shutil.move(result_dir / sub_dir, data_path)
        except shutil.Error:
            temp = result_dir / "temp"
            shutil.move(result_dir / sub_dir, temp)
            shutil.move(temp, data_path)

        all_files = [
            data_path / d_path / file
            for d_path in collect_paths(data_path)
            for file in os.listdir(d_path)
        ]
        sub_sub_dir = [
            data_path / element
            for element in os.listdir(data_path)
            if os.path.isdir(data_path / element)
        ]
        for file in all_files:
            shutil.move(file, data_path / file.name)
        for directory in sub_sub_dir:
            shutil.rmtree(directory)

    # rename data
    data_paths = [collect_leaf_paths(result_dir / sub_dir) for sub_dir in sub_dir_names]
    data_paths = [element for path_list in data_paths for element in path_list]
    for sub_dir in data_paths:
        for file in os.listdir(sub_dir):
            if file.endswith(".tif"):
                if file.startswith("t"):
                    new_name = file[1:]
                elif file.startswith("mask"):
                    new_name = file.replace("mask", "")
                elif file.startswith("man_track"):
                    new_name = file.replace("man_track", "")
                elif file.startswith("man_seg"):
                    new_name = file.replace("man_seg", "")
                else:
                    raise AssertionError(file)
                shutil.move(sub_dir / file, sub_dir / new_name)
            if file.endswith(".txt"):
                file_name = "lineage.txt"
                shutil.move(sub_dir / file, sub_dir.parent / file_name)
    for element in os.listdir(result_dir):
        if os.path.isdir(result_dir / element):
            if element not in sub_dir_names:
                shutil.rmtree(result_dir / element)


def generate_st_gt_lineage(st_seg_path, gt_track_path, res_path=None):
    """
    Use the ST segmentation and the GT tracking annotations to create an annotated dataset.
    Args:
        st_seg_path (string): path to the ST annotation
        gt_track_path (string): path to the GT tracking annotation
        res_path (string): path where to store the fused annotations

    Returns:

    """
    if res_path is None:
        res_path = st_seg_path

    st_seg_files = {
        int(re.findall("\d+", file)[0]): os.path.join(st_seg_path, file)
        for file in os.listdir(st_seg_path)
        if file.endswith(".tif")
    }
    gt_track_files = {
        int(re.findall("\d+", file)[0]): os.path.join(gt_track_path, file)
        for file in os.listdir(gt_track_path)
        if file.endswith(".tif")
    }
    lineage_file = [
        os.path.join(gt_track_path, file)
        for file in os.listdir(gt_track_path)
        if file.endswith(".txt")
    ][0]

    gt_lineage = pd.read_csv(
        lineage_file,
        delimiter=" ",
        header=None,
        names=["cell_id", "t_start", "t_end", "predecessor"],
    )
    gt_lineage = gt_lineage.set_index("cell_id")
    st_lineage = {}  # cellid: (t_start, t_end, successor)}

    t_max = gt_lineage["t_end"].max()
    st_seg_files = {
        time_point: img_path
        for time_point, img_path in st_seg_files.items()
        if time_point <= t_max
    }
    for time in sorted(st_seg_files.keys()):
        st_seg_mask = imread(st_seg_files[time])
        gt_track_mask = imread(gt_track_files[time])
        fused_mask = np.zeros_like(gt_track_mask)
        seg_mask_indices = get_indices_pandas(st_seg_mask)
        for mask_id, mask_indices in seg_mask_indices.items():
            gt_obj_ids = np.unique(gt_track_mask[mask_indices])
            gt_obj_ids = gt_obj_ids[gt_obj_ids > 0]
            if len(gt_obj_ids) == 1:  # use st mask as mapped to single obj
                gt_id = gt_obj_ids[0]
                fused_mask[mask_indices] = gt_id
                if gt_id not in st_lineage:
                    predecessor = gt_lineage.loc[gt_id]["predecessor"]
                    # since some gt obj might be not segmented by the st check if the predecessor exists
                    # - e.g. has already been added to the lineage
                    if predecessor in st_lineage:
                        st_lineage[gt_id] = [time, time, predecessor]
                    else:
                        st_lineage[gt_id] = [time, time, 0]
                else:
                    st_lineage[gt_id][1] = time
                # no gt mask == false positive -> don't add
        file_name = os.path.split(st_seg_files[time])[1]
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        imsave(os.path.join(res_path, file_name), fused_mask)
    df = pd.DataFrame.from_dict(st_lineage, orient="index")
    df = df.reset_index().sort_values("index")
    df.to_csv(
        os.path.join(res_path, "res_track.txt"), sep=" ", index=False, header=False
    )


def train_val_split(data_path, train_val_subdirs, val_split):
    """
    Split dataset in train/val by splitting from each image sequence a fixed percentage for validation.
    Args:
        data_path (Path): path to the data set
        train_val_subdirs (list): list of image sequences to use for training/validation
        val_split (float): fraction of images+masks to split from each image sequence for validation;
                            all remaining images+masks are used for training
    """
    for sub_dir in train_val_subdirs:
        data_path_train = data_path / "train" / sub_dir
        shutil.move(data_path / sub_dir, data_path_train)

        data_path_val = data_path / "val" / sub_dir
        if not os.path.exists(data_path_val):
            os.makedirs(data_path_val / "images")
            os.makedirs(data_path_val / "masks")
        # use last n frames for eval as CTC data is an image sequence
        images = [
            (int(image_file.split(".")[0]), image_file)
            for image_file in os.listdir(data_path_train / "images")
        ]
        masks = [
            (int(mask_file.split(".")[0]), mask_file)
            for mask_file in os.listdir(data_path_train / "masks")
        ]
        images.sort(key=lambda x: x[0])
        masks.sort(key=lambda x: x[0])
        # use masks since often more raw images than annotated images
        n_val = int(len(masks) * val_split)
        val_id_start = masks[-n_val][0]
        if not n_val > 0:
            return
        for img_id, img_file in images:
            if img_id < val_id_start:
                continue
            shutil.move(
                data_path_train / "images" / img_file,
                data_path_val / "images" / img_file,
            )
        for mask_id, mask_file in masks:
            if mask_id < val_id_start:
                continue
            shutil.move(
                data_path_train / "masks" / mask_file,
                data_path_val / "masks" / mask_file,
            )

        split_lineage_file_train_val(
            os.path.join(data_path_train, "lineage.txt"), val_split
        )


def prepare_ctc_data(
    source_path, result_path, keep_st=True, val_split=0.1, sub_dir_names=["01", "02"]
):
    """
    Copy CTC dataset and prepare a train/val split.
    Args:
        source_path (string): path where the original CTC data set is stored
        result_path (string): path where the prepared (train/val) data set will be stored
        keep_st (bool): use ST (silver truth) or GT (gold truth) annotation
        val_split (float): fraction of images+masks to split from each image sequence for validation;
                            all remaining images+masks are used for training
        val_split (float):
        sub_dir_names (list): list of image sequences to use for training/validation

    Returns:

    """
    source_path = Path(source_path)
    result_path = Path(result_path)
    result_path = result_path / source_path.name
    copy_and_rename_data(source_path, result_path, keep_st, sub_dir_names)
    train_val_split(result_path, sub_dir_names, val_split)


def prepare_ctc_gt(source_path, result_path, val_split=0.1, sub_dir_names=["01", "02"]):
    """
    Prepare the GT ctc data set with the same train/val split
    Args:
        source_path (string): path where the original CTC data set is stored
        result_path (string): path where the prepared (train/val) gt data set will be stored
        val_split (float): fraction of images+masks to split from each image sequence for validation;
                            all remaining images+masks are used for training
        sub_dir_names:  list of image sequences to use for training/validation
    """
    res_path = os.path.join(result_path, Path(source_path).name, "gt")
    os.makedirs(os.path.join(res_path, "train"))
    os.makedirs(os.path.join(res_path, "val"))

    for sub_dir in os.listdir(source_path):
        if not sub_dir.endswith("_GT"):
            continue
        sub_dir_name = sub_dir.split("_")[0]
        assert sub_dir_name.isnumeric(), f"unknown directory naming {sub_dir}"
        if sub_dir_name in sub_dir_names:
            data_path = Path(os.path.join(res_path, "train", sub_dir))

            shutil.copytree(
                os.path.join(source_path, sub_dir),
                data_path,
            )
            split_gt_data(data_path, val_split)


def split_lineage_file_train_val(lineage_file, val_split):
    """
    Split lineage file in train and val lineage.
    Args:
        lineage_file  (string): path to the lineage file
        val_split (float): fraction of images+masks to split from each image sequence for validation;
                            all remaining images+masks are used for training
    """
    lineage_name = os.path.basename(lineage_file)
    full_lineage = pd.read_csv(
        lineage_file,
        delimiter=" ",
        header=None,
        names=["cell_id", "t_start", "t_end", "predecessor"],
    )
    time_points = np.arange(
        full_lineage["t_start"].min(), full_lineage["t_end"].max() + 1
    )
    n_val = int(len(time_points) * val_split)
    t_val_start = time_points[-n_val]

    train_lineage = full_lineage.copy()
    t_train_end = time_points[-n_val - 1]
    train_lineage = train_lineage[train_lineage["t_start"] <= t_train_end]
    train_lineage["t_end"] = train_lineage["t_end"].apply(lambda x: min(t_train_end, x))
    train_lineage.to_csv(lineage_file, sep=" ", header=False, index=False)

    val_lineage = full_lineage.copy()
    val_lineage = val_lineage[val_lineage["t_end"] >= t_val_start]
    val_lineage["t_start"] = val_lineage["t_start"].apply(lambda x: max(t_val_start, x))
    cells_val_data = val_lineage["cell_id"].values
    val_lineage["predecessor"] = val_lineage["predecessor"].apply(
        lambda x: 0 if x not in cells_val_data else x
    )
    # navigate to path: .../DATA_SET/ from .../DATA_SET/train/SEQUENCE_ID
    d_path = Path(lineage_file).parent.parent.parent
    img_sequence = Path(lineage_file).parent.name
    val_lineage.to_csv(
        os.path.join(d_path, "val", img_sequence, lineage_name),
        sep=" ",
        header=False,
        index=False,
    )


def split_gt_data(res_path, val_split):
    """
    Split gt (ground truth) data into train/val data sets.
    Args:
        res_path (Path): path to the gt dataset
        val_split (float): fraction of images+masks to split from each image sequence for validation;
                        all remaining images+masks are used for training
    """
    tra_path = res_path / "TRA"
    time_points = [
        int(re.findall(r"\d+", mask_file)[0])
        for mask_file in os.listdir(tra_path)
        if mask_file.endswith("tif")
    ]
    time_points.sort(key=lambda x: x)
    n_val = int(len(time_points) * val_split)
    t_val_start = time_points[-n_val]
    for seg_track_gt_path in collect_leaf_paths(os.path.join(res_path)):
        mask_files = [
            (int(re.findall(r"\d+", file)[0]), file)
            for file in os.listdir(seg_track_gt_path)
            if file.endswith("tif")
        ]
        mask_files = filter(lambda x: x[0] >= t_val_start, mask_files)
        path_parts = Path(seg_track_gt_path).parts
        val_path = os.path.join(*path_parts[:-3], "val", *path_parts[-2:])
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        for _, mask_file in mask_files:
            shutil.move(
                os.path.join(seg_track_gt_path, mask_file),
                os.path.join(val_path, mask_file),
            )
    lineage_file = os.path.join(tra_path, "man_track.txt")
    gt_lineage = pd.read_csv(
        lineage_file,
        delimiter=" ",
        header=None,
        names=["cell_id", "t_start", "t_end", "predecessor"],
    )

    train_lineage = gt_lineage.copy()
    t_train_end = time_points[-n_val - 1]
    train_lineage = train_lineage[train_lineage["t_start"] <= t_train_end]
    train_lineage["t_end"] = train_lineage["t_end"].apply(lambda x: min(t_train_end, x))
    train_lineage.to_csv(lineage_file, sep=" ", header=False, index=False)

    val_lineage = gt_lineage.copy()
    val_lineage = val_lineage[val_lineage["t_end"] >= t_val_start]
    val_lineage["t_start"] = val_lineage["t_start"].apply(lambda x: max(t_val_start, x))
    cells_val_data = val_lineage["cell_id"].values
    val_lineage["predecessor"] = val_lineage["predecessor"].apply(
        lambda x: 0 if x not in cells_val_data else x
    )
    path_parts = Path(lineage_file).parts
    val_path = os.path.join(*path_parts[:-4], "val", *path_parts[-3:])
    val_lineage.to_csv(val_path, sep=" ", header=False, index=False)
