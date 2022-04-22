"""
Original work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
parts of generate_image_crops and init_training based on code of Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)

"""
import json
from datetime import datetime
from glob import glob

from PIL import Image
from tqdm import tqdm

from embedtrack.utils.create_dicts import (
    create_dataset_dict,
    create_model_dict,
    create_loss_dict,
    create_configs,
)

from embedtrack.datasets.generate_crops import *
from embedtrack.datasets.prepare_data import prepare_ctc_data
from embedtrack.datasets.prepare_data import prepare_ctc_gt
from embedtrack.train.train import begin_training
import pandas as pd
import glob

IMAGE_FORMATS = ("bmp", "jpeg", "tif", "png", "tiff")
MODES_TO_BITS = {
    "1": 1,
    "L": 8,
    "P": 8,
    "RGB": 8,
    "RGBA": 8,
    "CMYK": 8,
    "YCbCr": 8,
    "I": 32,
    "F": 32,
    "I;16": 16,
    "I;16B": 16,
    "I;16N": 16,
    "BGR;15": 15,
    "BGR;24": 24,
}


class DataConfig:
    def __init__(
        self,
        raw_data_path,
        data_set,
        data_path,
        use_silver_truth=False,
        train_val_sequences=["01", "02"],
        train_val_split=0.2,
    ):
        """
        Configuration of the training and vaildation dataset
        Args:
            raw_data_path (string): Path where the CTC datasets are stored
            data_set (string): Name of the dataset
            data_path (string): Path where to store to the prepared data for training the model
            use_silver_truth (bool): Use the ST from the cell tracking challenge or use the GT annotations
            train_val_sequences (list): list of the image sequences to use for training and validation
            train_val_split (float): fraction of images to split from each image sequence provided in train_val_sequences for validation
        """
        self.raw_data_path = raw_data_path
        self.data_set = data_set
        self.data_path = data_path
        self.use_silver_truth = use_silver_truth
        self.train_val_sequences = train_val_sequences
        self.train_val_split = train_val_split


class ModelConfig:
    def __init__(self, input_channels=1, n_seg_classes=[4, 1], n_track_classes=2):
        """
        Configuration of the EmbedTrack model
        Args:
            input_channels (int): number of input channels of the raw images
            n_seg_classes (list): number of output channels of the two segmentation decoders
            n_track_classes (int): number of output channels of the tracking decoder (2D images: 2, 3D images: 3)
        """
        self.input_channels = input_channels
        self.n_seg_classes = n_seg_classes
        self.n_track_classes = n_track_classes


class TrainConfig:
    def __init__(
        self,
        save_model_path,
        crop_size=256,
        center="medioid",
        resume_training=False,
        train_size=3000,
        train_batch_size=16,
        virtual_train_batch_multiplier=1,
        val_size=2600,
        val_batch_size=16,
        virtual_val_batch_multiplier=1,
        n_epochs=200,
        display=True,
        display_it=50,
        learning_rate=5e-4,
    ):
        """
        Configuration of the Training
        Args:
            save_model_path (string): path where to store the trained models
            crop_size (int): size of the quadratic image crops the model is trained on
            center (string): indicates how the cell center is calculated (options: "centroid", "approximate-medoid", "medoid")
            resume_training (bool): continue training from a checkpoint
            train_size (int, optional): size of the dataset to train on if None - the full dataset is used per epoch if an int is provided per epoch only a subset of the training data is used
            train_batch_size (int): batch size during training
            virtual_train_batch_multiplier (int): increase batch size virtually by applying and optim step after every N batches (the loss is averages to loss/N per batch)
            val_size(int, optional): size of the dataset to val on if None - the full dataset is used per epoch if an int is provided per epoch only a subset of the validation data is used
            val_batch_size (int): batch size during training
            virtual_val_batch_multiplier(int): increase batch size virtually by averaging the loss to loss/N per batch
            n_epochs (int): max number of epochs to train the model
            display (bool): display visualization of the training
            display_it (int): display visualization every N iterations if display is True
            learning_rate (float): learning rate of the optimizer
        """
        self.save_model_path = save_model_path
        self.crop_size = crop_size
        self.center = center  # 'centroid', 'approximate-medoid', 'medoid'
        self.resume_training = resume_training
        self.train_size = train_size
        self.train_batch_size = train_batch_size
        self.virtual_train_batch_multiplier = virtual_train_batch_multiplier
        self.val_size = val_size
        self.val_batch_size = val_batch_size
        self.virtual_val_batch_multiplier = virtual_val_batch_multiplier
        self.n_epochs = n_epochs  # todo: add early stopping
        self.display = display
        self.display_it = display_it
        self.learning_rate = learning_rate


def copy_ctc_data(data_config):
    """
    Copy CTC data according to the data_config
    Args:
        data_config (DataConfig): instance of DataConfig providing the data set, source and destination path as well as the train/val split

    """
    if os.path.exists(os.path.join(data_config.data_path, data_config.data_set)):
        print(
            f"{os.path.join(data_config.data_path, data_config.data_set)} already exists,"
            f" therefore no data is copied from {os.path.join(data_config.raw_data_path, data_config.data_set)}"
        )
    else:
        print(f"prepare data of {data_config.data_set}")
        prepare_ctc_data(
            os.path.join(data_config.raw_data_path, data_config.data_set),
            data_config.data_path,
            keep_st=data_config.use_silver_truth,
            val_split=data_config.train_val_split,
            sub_dir_names=data_config.train_val_sequences,
        )
        prepare_ctc_gt(
            os.path.join(data_config.raw_data_path, data_config.data_set),
            data_config.data_path,
            val_split=data_config.train_val_split,
            sub_dir_names=data_config.train_val_sequences,
        )
        print(f"data stored in {data_config.data_path}")


def generate_image_crops(data_config, train_config, model_config):
    """
    Generate image crops for training and evaluation
    Args:
        data_config (DataConfig): instance of DataConfig providing info on data path, train/val split
        train_config (TrainConfig): instance of TrainConfig providing config for the training process
        model_config:(ModelConfig): instance of ModelConfig providing config for the network

    Returns:

    """
    crops_dir = os.path.join(data_config.data_path, "crops")

    if os.path.exists(os.path.join(crops_dir, data_config.data_set)):
        print(
            f"{os.path.join(crops_dir, data_config.data_set)} already exists, therefore no crops are generated from {os.path.join(data_config.data_path, data_config.data_set)}"
        )
    else:

        img_sequences = list(
            os.listdir(
                os.path.join(
                    data_config.data_path,
                    data_config.data_set,
                    "train",
                )
            )
        )
        d_path = os.path.join(
            data_config.data_path,
            data_config.data_set,
            "train",
            img_sequences[0],
            "images",
        )
        img_files = (
            os.path.join(d_path, element)
            for element in os.listdir(d_path)
            if element.endswith(IMAGE_FORMATS)
        )
        # decode pixel depth and image dimension
        with Image.open(next(img_files)) as img:
            # fixme: case RGB image
            pix_depth = MODES_TO_BITS[img.mode]
            is_multichannel_img = pix_depth == 8 and len(img.mode) > 1
            if is_multichannel_img:
                print(
                    f"Multichannel image detected. Image shape is f{img.size} - compare number of image channels with number of model input channels f{model_config.input_channels}"
                )
                if len(img.size) == 3:
                    img_dim = "2d"
                elif len(img.size) == 4:
                    img_dim = "3d"
                else:
                    raise AssertionError(f"Unknown image dimensions {img.size}")
            else:
                if len(img.size) == 2:
                    img_dim = "2d"
                elif len(img.size) == 3:
                    img_dim = "3d"
                else:
                    raise AssertionError(f"Unknown image dimensions {img.size}")
        data_sub_sets = [
            "/".join([data_split, img_sequence])
            for data_split in ["train", "val"]
            for img_sequence in os.listdir(
                os.path.join(data_config.data_path, data_config.data_set, data_split)
            )
        ]
        data_properties_dir = dict()
        if isinstance(train_config.crop_size, int):
            grid_shape = (train_config.crop_size, train_config.crop_size)

        elif len(train_config.crop_size) == 2:
            grid_shape = (train_config.crop_size[0], train_config.crop_size[1])
        else:
            raise AssertionError(f"Unknown crop size {train_config.crop_size}")

        data_properties_dir["n_y"], data_properties_dir["n_x"] = grid_shape
        data_properties_dir["data_type"] = str(pix_depth) + "-bit"

        with open(
            os.path.join(
                data_config.data_path, data_config.data_set, "data_properties.json"
            ),
            "w",
        ) as outfile:
            json.dump(data_properties_dir, outfile)
            print(
                f"Dataset properies of the `{data_config.data_set}` dataset is saved to {os.path.join(data_config.data_path, data_config.data_set,'data_properties.json')}"
            )

        print(crops_dir)
        for data_subset in data_sub_sets:
            image_dir = os.path.join(
                data_config.data_path, data_config.data_set, data_subset, "images"
            )
            instance_dir = os.path.join(
                data_config.data_path, data_config.data_set, data_subset, "masks"
            )
            image_names = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
            instance_names = sorted(glob.glob(os.path.join(instance_dir, "*.tif")))
            for i in tqdm(np.arange(len(image_names))):
                # create img crops only if segmentation mask exists
                try:
                    idx_segm_mask = instance_names.index(
                        os.path.join(instance_dir, os.path.basename(image_names[i]))
                    )
                except ValueError:
                    continue
                # generate crops for an image
                process(
                    image_names[i],
                    instance_names[idx_segm_mask],
                    os.path.join(crops_dir, data_config.data_set),
                    data_subset,
                    train_config.crop_size,
                    train_config.center,
                )
            print(
                "Cropping of images, instances and centre_images for data_subset = `{}` done!".format(
                    data_subset
                )
            )
        # generate offset images (shift between two cell centers between successive frames) for tracking
        for data_subset in data_sub_sets:
            instance_dir = os.path.join(
                data_config.data_path, data_config.data_set, data_subset, "masks"
            )
            center_image_path = os.path.join(
                data_config.data_path,
                data_config.data_set,
                data_subset,
                f"center-{train_config.center}",
            )
            lineage_file = os.path.join(
                data_config.data_path,
                data_config.data_set,
                data_subset,
                "lineage.txt",
            )
            lineage_data = pd.read_csv(
                lineage_file,
                delimiter=" ",
                header=None,
                names=["cell_id", "t_start", "t_end", "predecessor"],
            )
            lineage = {
                cell["cell_id"]: cell["predecessor"]
                for _, cell in lineage_data.iterrows()
            }

            calc_obj_shifts(
                instance_dir,
                center_image_path,
                os.path.join(crops_dir, data_config.data_set, data_subset),
                lineage,
                train_config.crop_size,
            )
            # remove empty crops
            offset_path = os.path.join(
                os.path.join(crops_dir, data_config.data_set, data_subset),
                (Path(center_image_path).name + "-" + "flow"),
            )

            centroid_path = os.path.join(
                os.path.join(crops_dir, data_config.data_set, data_subset),
                (Path(center_image_path).name),
            )
            image_path = os.path.join(
                crops_dir, data_config.data_set, data_subset, "images"
            )
            instances_path = os.path.join(
                crops_dir, data_config.data_set, data_subset, "masks"
            )
            data_files, img_pairs = get_image_pairs(
                image_path, instances_path, offset_path, centroid_path
            )
            remove_empty_crops(*data_files, img_pairs)


def get_image_pairs(img_path, instance_path, offset_path, centroid_path):
    """
    Get all pairs of succssive frames.
    Args:
        img_path (string):path to the raw image files
        instance_path (string) : path to the segmentation mask files
        offset_path (string): path to the offset image files
        centroid_path (string): path to the centroid image files

    Returns: (tuple, list): tuple of lists containing all paths to the different image files; list of pairs of successive frames

    """
    pairs = []
    image_files, instance_files, centroid_files = list(
        zip(
            *[
                (
                    os.path.join(img_path, img_file),
                    os.path.join(instance_path, img_file),
                    os.path.join(centroid_path, img_file),
                )
                for img_file in os.listdir(img_path)
            ]
        )
    )
    offset_files = [os.path.join(offset_path, file) for file in os.listdir(offset_path)]
    for i, path_img_file in enumerate(image_files):
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
        path_flow_img = os.path.join(offset_path, flow_img_name)
        try:
            pairs.append(
                (
                    image_files.index(os.path.join(path_img, name_next_img)),
                    i,
                    offset_files.index(path_flow_img),
                )
            )
        except ValueError:
            continue
    return (image_files, instance_files, offset_files, centroid_files), pairs


def remove_empty_crops(
    image_files, instance_files, offset_files, centroid_files, img_pairs
):
    """
    Remove empty image crops from the train/val data set.
    Args:
        image_files (list): list of raw image files
        instance_files (list):list of mask image files
        offset_files (list): list of offset image files
        centroid_files (list): list of centroid image files
        img_pairs (list): list of successive image frames

    Returns:

    """

    for pair_idx in img_pairs:
        img_idx_curr, img_idx_prev, offset_idx = pair_idx
        try:
            instances_curr = tifffile.imread(instance_files[img_idx_curr])
        except FileNotFoundError:
            instances_curr = np.array([0])
        try:
            instances_prev = tifffile.imread(instance_files[img_idx_prev])
        except FileNotFoundError:
            instances_prev = np.array([0])

        mask_ids = np.unique(
            np.concatenate([instances_curr.reshape(-1), instances_prev.reshape(-1)])
        )
        mask_ids = mask_ids[mask_ids != 0]
        if len(mask_ids) == 0:
            delete_image_crop(instance_files[img_idx_curr])
            delete_image_crop(image_files[img_idx_curr])
            delete_image_crop(centroid_files[img_idx_curr])
            delete_image_crop(instance_files[img_idx_prev])
            delete_image_crop(image_files[img_idx_prev])
            delete_image_crop(centroid_files[img_idx_prev])
            delete_image_crop(offset_files[offset_idx])


def delete_image_crop(img_file):
    try:
        os.remove(img_file)
    except FileNotFoundError:
        pass


def init_training(data_config, train_config, model_config):
    """
    Initialize training of the model.
    Args:
         data_config (DataConfig): instance of DataConfig providing info on data path, train/val split
        train_config (TrainConfig): instance of TrainConfig providing config for the training process
        model_config:(ModelConfig): instance of ModelConfig providing config for the network
    """
    crops_dir = os.path.join(data_config.data_path, "crops")
    if not os.path.exists(
        os.path.join(
            data_config.data_path, data_config.data_set, "data_properties.json"
        )
    ):
        data_properties_file = os.path.join(
            data_config.data_path, data_config.data_set, "data_properties.json"
        )
        raise AssertionError(f"No such file f{data_properties_file}")
    with open(
        os.path.join(
            data_config.data_path, data_config.data_set, "data_properties.json"
        )
    ) as json_file:
        data = json.load(json_file)
        data_type, n_y, n_x = (
            data["data_type"],
            int(data["n_y"]),
            int(data["n_x"]),
        )

    train_subsets = [
        "/".join(["train", img_sequence])
        for img_sequence in os.listdir(
            os.path.join(data_config.data_path, data_config.data_set, "train")
        )
        if img_sequence in data_config.train_val_sequences
    ]
    val_subsets = [
        "/".join(["val", img_sequence])
        for img_sequence in os.listdir(
            os.path.join(data_config.data_path, data_config.data_set, "val")
        )
        if img_sequence in data_config.train_val_sequences
    ]
    train_dataset_dict = create_dataset_dict(
        data_dir=data_config.data_path,
        crop_dir=crops_dir,
        data_subsets=train_subsets,
        project_name=data_config.data_set,
        center=train_config.center,
        size=train_config.train_size,
        batch_size=train_config.train_batch_size,
        virtual_batch_multiplier=train_config.virtual_train_batch_multiplier,
        type="train",
        workers=8,
    )

    val_dataset_dict = create_dataset_dict(
        data_dir=data_config.data_path,
        crop_dir=crops_dir,
        data_subsets=val_subsets,
        project_name=data_config.data_set,
        center=train_config.center,
        size=train_config.val_size,
        batch_size=train_config.val_batch_size,
        virtual_batch_multiplier=train_config.virtual_val_batch_multiplier,
        type="val",
        workers=8,
    )

    n_classes = [*model_config.n_seg_classes, model_config.n_track_classes]
    model_dict = create_model_dict(
        input_channels=model_config.input_channels,
        n_classes=n_classes,
    )
    loss_dict = create_loss_dict()

    if train_config.resume_training:
        resume_path = (
            Path(train_config.save_model_path) / "best_iou_model.pth"
        ).as_posix()

        save_dir = train_config.save_model_path
    else:
        resume_path = ""
        save_dir = os.path.join(
            train_config.save_model_path, datetime.now().strftime("%Y-%m-%d---%H-%M-%S")
        )
    # todo: setting n_x, n_y for all grids!!
    configs = create_configs(
        n_epochs=train_config.n_epochs,
        display=train_config.display,
        display_it=train_config.display_it,
        resume_path=resume_path,
        save_dir=save_dir,
        grid_y=n_y,
        grid_x=n_x,
        train_lr=train_config.learning_rate,
    )

    # todo: write json with all settings:
    # loss
    # model dir
    # data dir
    # train dir
    # configs
    min_mask_size = calc_min_mask_size(
        [
            os.path.join(data_config.data_path, data_config.data_set, train_p)
            for train_p in train_subsets
        ]
    )
    grid_config = {
        key: configs[key] for key in ["grid_x", "grid_y", "pixel_x", "pixel_y"]
    }
    train_dict = vars(train_config).copy()
    train_dict.update(min_mask_size=min_mask_size)
    all_configs = {
        "loss_dict": loss_dict,
        "model_dict": model_dict,
        "data_dict": vars(data_config),
        "train_dict": train_dict,
        "grid_dict": grid_config,
    }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "config.json"), "w") as json_file:
        print(json.dumps(all_configs, sort_keys=True, indent=4), file=json_file)

    begin_training(
        train_dataset_dict,
        val_dataset_dict,
        model_dict,
        loss_dict,
        configs,
    )


def calc_min_mask_size(train_paths):
    """Calculate the 1% percentile of the mask sizes from the train data set."""
    all_mask_sizes = []
    for train_p in train_paths:
        for file in os.listdir(os.path.join(train_p, "masks")):
            if file.endswith(".tif"):
                segm_mask = tifffile.imread(os.path.join(train_p, "masks", file))
                masks = get_indices_pandas(segm_mask)
                if len(masks) > 0:
                    mask_sizes = masks.apply(lambda x: len(x[0])).values
                    all_mask_sizes.extend(mask_sizes)
    return np.percentile(all_mask_sizes, 1)


def run_pipeline(data_config, train_config, model_config):
    #######################
    #     copy data       #
    #######################
    copy_ctc_data(data_config)
    #######################
    # generate crops      #
    #######################
    generate_image_crops(data_config, train_config, model_config)
    #######################
    #      training       #
    #######################
    init_training(data_config, train_config, model_config)
