"""
Original work Copyright 2019 Davy Neven,  KU Leuven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))
Modified work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
Modifications: remove normalization functions; edit crop generation; calculate offsets between two frames; load image pairs
"""
import os
from itertools import product
from pathlib import Path
import numpy as np
import tifffile
from numba import jit
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes

from embedtrack.utils.utils import get_indices_pandas


def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in set(np.unique(lbl_img)) - set([0]):
        mask = lbl_img == l
        mask_filled = binary_fill_holes(mask, **kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def fill_label_holes(lbl_img, **kwargs):
    """
    Fill small holes in label image.
    """

    def grow(sl, interior):
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior)
        )

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


@jit(nopython=True)
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D


def generate_center_image(instance, center, ids):
    """
    Generates a `center_image` which is one (True) for all center locations and zero (False) otherwise.
    Parameters
    ----------
    instance: numpy array
        `instance` image containing unique `ids` for each object (YX)
         or present in a one-hot encoded style where each object is one in it own slice and zero elsewhere.
    center: string
        One of 'centroid', 'approximate-medoid' or 'medoid'.
    ids: list
        Unique ids corresponding to the objects present in the instance image.
        True (in this case, `instance` has shape DYX) or False (in this case, `instance` has shape YX).
    """

    center_image = np.zeros((instance.shape[-2], instance.shape[-1]), dtype=bool)
    for j, id in enumerate(ids):
        y, x = np.where(instance == id)
        if len(y) != 0 and len(x) != 0:
            if center == "centroid":
                ym, xm = np.mean(y), np.mean(x)
            elif center == "approximate-medoid":
                ym_temp, xm_temp = np.median(y), np.median(x)
                imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2)
                ym, xm = y[imin], x[imin]
            elif center == "medoid":
                ### option - 1 (scipy `distance_matrix`) (slow-ish)
                # dist_matrix = distance_matrix(np.vstack((x, y)).transpose(), np.vstack((x, y)).transpose())
                # imin = np.argmin(np.sum(dist_matrix, axis=0))
                # ym, xm = y[imin], x[imin]

                ### option - 2 (`hdmedoid`) (slightly faster than scipy `distance_matrix`)
                # ym, xm = hd.medoid(np.vstack((y,x)))

                ### option - 3 (`numba`)
                dist_matrix = pairwise_python(np.vstack((x, y)).transpose())
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                ym, xm = y[imin], x[imin]
            center_image[int(np.round(ym)), int(np.round(xm))] = True
    return center_image


def process(
    im,
    inst,
    crops_dir,
    data_subset,
    crop_size,
    center,
    overlap=0.3,
):
    """
    Processes the actual images and instances to generate crops of size `crop-size`.
    Additionally, one could perform min-max normalization of the crops at this stage (False, by default)
    Parameters
    ----------
    im: numpy array
        Raw image which must be processed (segmented)
    inst: numpy array
        Corresponding instance mask which contains objects identified by their unique ids (YX)
    crops_dir: string
        Indicates the path where the crops should be saved
    center: string
        One of `centroid`, `approximate-medoid` or `medoid`
    """
    center_img_path = os.path.join(Path(im).parent.parent, ("center-" + center + "/"))
    img_crop_path = os.path.join(crops_dir, data_subset, "images/")
    instance_crop_path = os.path.join(crops_dir, data_subset, "masks/")
    center_crop_img_path = os.path.join(
        crops_dir, data_subset, "center-" + center + "/"
    )

    if not os.path.exists(img_crop_path):
        os.makedirs(os.path.dirname(img_crop_path))
        print("Created new directory : {}".format(img_crop_path))
    if not os.path.exists(instance_crop_path):
        os.makedirs(os.path.dirname(instance_crop_path))
        print("Created new directory : {}".format(instance_crop_path))
    if not os.path.exists(center_crop_img_path):
        os.makedirs(os.path.dirname(center_crop_img_path))
        print("Created new directory : {}".format(center_crop_img_path))
    if not os.path.exists(center_img_path):
        os.makedirs(os.path.dirname(center_img_path))
        print("Created new directory : {}".format(center_img_path))

    instance = tifffile.imread(inst).astype(np.uint16)
    image = tifffile.imread(im)
    instance = fill_label_holes(instance)

    size_x, size_y = image.shape
    n_max_x = int(size_x / int(crop_size * overlap))
    n_max_y = int(size_y / int(crop_size * overlap))
    x_start = (np.arange(0, n_max_x) * crop_size * overlap).astype(np.int)
    y_start = (np.arange(0, n_max_y) * crop_size * overlap).astype(np.int)
    upper_left = product(x_start, y_start)
    n_ids = np.unique(instance)
    n_ids = n_ids[n_ids > 0]
    center_img = generate_center_image(instance, center, n_ids)
    tifffile.imsave(
        center_img_path + os.path.basename(im)[:-4] + ".tif",
        center_img,
    )
    for i, (x, y) in enumerate(upper_left):
        im_crop = image[x : x + crop_size, y : y + crop_size]
        if np.any(np.array(im_crop.shape) < crop_size):
            continue
        instance_crop = instance[x : x + crop_size, y : y + crop_size]
        center_image_crop = center_img[x : x + crop_size, y : y + crop_size]
        tifffile.imsave(
            img_crop_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(i),
            im_crop,
        )
        tifffile.imsave(
            instance_crop_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(i),
            instance_crop,
        )
        tifffile.imsave(
            center_crop_img_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(i),
            center_image_crop,
        )


def calc_obj_shifts(
    instance_path, center_img_path, crops_path, lineage, crop_size, overlap=0.3
):
    """
    Calculate for each cell at t its offset to its corresponding cell center at t-1
    Args:
        instance_path:  string
            Path to the segmentation masks
        center_img_path: string
            Path to the center image masks
        crops_path: str
            Path where to store the generated crops
        lineage: dict
            Predecessor information for each track
        crop_size: int
            Size of the squared shaped crops
        overlap: float
            overlap between neighbouring crops


    Returns:

    """
    center_image_list = [
        os.path.join(center_img_path, file_name)
        for file_name in os.listdir(center_img_path)
        if file_name.endswith(".tif")
    ]
    center_image_list.sort()
    obj_shift_crops_path = os.path.join(
        crops_path, (Path(center_img_path).name + "-" + "flow")
    )
    obj_shift_path = os.path.join(
        Path(instance_path).parent, (Path(center_img_path).name + "-" + "flow")
    )
    if not os.path.exists(obj_shift_crops_path):
        os.makedirs(obj_shift_crops_path)
        print("Created new directory : {}".format(obj_shift_crops_path))
    if not os.path.exists(obj_shift_path):
        os.makedirs(obj_shift_path)
        print("Created new directory : {}".format(obj_shift_path))

    image_pairs_index = get_image_pairs(center_image_list)

    size_x, size_y = tifffile.imread(center_image_list[0]).shape
    n_max_x = int(size_x / int(crop_size * overlap))
    n_max_y = int(size_y / int(crop_size * overlap))
    x_start = (np.arange(0, n_max_x) * crop_size * overlap).astype(np.int)
    y_start = (np.arange(0, n_max_y) * crop_size * overlap).astype(np.int)
    for (index_curr, index_prev) in image_pairs_index:
        center_img_curr = tifffile.imread(center_image_list[index_curr])
        center_img_prev = tifffile.imread(center_image_list[index_prev])
        _, img_curr_name = os.path.split(center_image_list[index_curr])
        _, img_prev_name = os.path.split(center_image_list[index_prev])
        time_id_curr, file_ending = img_curr_name.split(".")
        time_id_prev = img_prev_name.split(".")[0]
        img_name = "_".join([time_id_curr, time_id_prev])
        flow = calc_center_diffs(
            tifffile.imread(os.path.join(instance_path, img_curr_name)),
            center_img_curr,
            tifffile.imread(os.path.join(instance_path, img_prev_name)),
            center_img_prev,
            lineage,
        )
        tifffile.imsave(
            os.path.join(obj_shift_path, img_name + "." + file_ending), flow
        )
        # generate crops

        for i, (x, y) in enumerate(product(x_start, y_start)):
            flow_img_crop = flow[:, x : x + crop_size, y : y + crop_size]
            if np.any(np.array(flow_img_crop.shape[1:]) < crop_size):
                continue

            tifffile.imsave(
                os.path.join(obj_shift_crops_path, img_name + "_{:03d}.tif".format(i)),
                flow_img_crop,
            )


def calc_center_diffs(
    instance_crop_curr, center_img_curr, instance_crop_prev, center_img_prev, lineage
):
    """
    Calculate for each pair of image crops t,t-1 the offset between the cell center at t to
    the center of its predecessor in t-1
    Args:
        instance_crop_curr: numpy array
            a segmentation mask crop at time t
        center_img_curr:  numpy array
            a crop of the calculated object center map at time t
        instance_crop_prev: numpy array
            a segmentation mask crop at time t-1
        center_img_prev: numpy array
            a crop of the calculated object center map at time t-1
        lineage: dict
            Predecessor information for each track

    Returns: numpy array
        array of offsets

    """
    seeds_curr = np.zeros_like(instance_crop_curr)
    seeds_curr[center_img_curr] = instance_crop_curr[center_img_curr]
    seeds_prev = np.zeros_like(instance_crop_prev)
    seeds_prev[center_img_prev] = instance_crop_prev[center_img_prev]
    object_ids_curr = get_indices_pandas(seeds_curr)
    object_ids_prev = get_indices_pandas(seeds_prev)
    flow = np.zeros(
        (*center_img_curr.shape, len(center_img_curr.shape))
    )  # hxwx2 or dxhxwx3
    for obj_id, seed_position in object_ids_curr.items():
        if obj_id in object_ids_prev:
            # same cell at t
            prev_position = object_ids_prev[obj_id]
        elif lineage[obj_id] in object_ids_prev:
            # position of mother cell at t
            prev_position = object_ids_prev[lineage[obj_id]]
        else:
            # cell starts at t+1
            continue

        difference = np.mean(np.stack(seed_position) - np.stack(prev_position), axis=-1)
        # shift all pixels by avg offset
        flow[tuple(seed_position[0]), tuple(seed_position[1]), :] = difference.reshape(
            1, -1
        )
    return np.transpose(flow, (2, 0, 1)).copy()  # cxhxw or cxdxhxw


def get_image_pairs(image_list):
    """
    Extract successive image pairs.
    Args:
        image_list: list
            list of image files
    Returns:

    """
    # same name for all crops: masks, images, center_images
    pairs = []
    for i, path_img_file in enumerate(image_list):
        path_img, name_img = os.path.split(path_img_file)
        time, ending = name_img.split(".")
        name_next_img = str(int(time) + 1).zfill(len(time)) + "." + ending
        try:
            # to assign daughters to mother object
            pairs.append((image_list.index(os.path.join(path_img, name_next_img)), i))
        except ValueError:
            continue
    return pairs
