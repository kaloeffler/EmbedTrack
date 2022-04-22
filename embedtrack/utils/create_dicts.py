"""
Original work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
Modifications: remove 3d; remove one hot; change augmentation; change create_model_dict
"""
import torch
import os
import embedtrack.utils.transforms as my_transforms


def create_dataset_dict(
    data_dir,
    crop_dir,
    data_subsets,
    project_name,
    size,
    center,
    type,
    name="2d",
    batch_size=16,
    virtual_batch_multiplier=1,
    workers=8,
):
    """
    Creates `dataset_dict` dictionary from parameters.
    Parameters
    ----------
    data_dir: string
        Data is read from os.path.join(data_dir, project_name)
    crop_dir: string
        Cropped data is read from os.path.join(crop_dir, project_name)
    data_subsets: list
        list of sub_directories in data_dir and crop_dir from which data will be used
    project_name: string
        Data is read from os.path.join(data_dir, project_name)
    size: int
        Number of image-mask per epoch
    center: string
        One of 'centroid', 'approximate-medoid', 'medoid'
    type: string
        One of 'train', 'val'
    name: string
        only "2d" currently
    batch_size: int
        Effective Batch-size is the product of `batch_size` and `virtual_batch_multiplier`
    virtual_batch_multiplier: int
        Effective Batch-size is the product of `batch_size` and `virtual_batch_multiplier`
    workers: int
        Number of data-loader workers
    """
    if name == "2d":
        set_transforms = my_transforms.get_transform(
            [
                {
                    "name": "RandomRotationsAndFlips",
                    "opts": {
                        "keys": (
                            "image_curr",
                            "image_prev",
                            "instance_curr",
                            "instance_prev",
                            "label_curr",
                            "label_prev",
                            "center_image_curr",
                            "center_image_prev",
                            "flow",
                        ),
                        "degrees": 90,
                    },
                },
                {
                    "name": "ImageAugmentation",
                    "opts": {
                        "keys": (
                            "image_curr",
                            "image_prev",
                            "instance_curr",
                            "instance_prev",
                            "label_curr",
                            "label_prev",
                            "center_image_curr",
                            "center_image_prev",
                            "flow",
                        ),
                    },
                },
                {
                    "name": "ToTensorFromNumpy",
                    "opts": {
                        "keys": (
                            "image_curr",
                            "image_prev",
                            "instance_curr",
                            "instance_prev",
                            "label_curr",
                            "label_prev",
                            "center_image_curr",
                            "center_image_prev",
                            "flow",
                        ),
                        "type": (
                            torch.FloatTensor,
                            torch.FloatTensor,
                            torch.ShortTensor,
                            torch.ShortTensor,
                            torch.ShortTensor,
                            torch.ShortTensor,
                            torch.BoolTensor,
                            torch.BoolTensor,
                            torch.FloatTensor,
                        ),
                    },
                },
            ]
        )
    else:
        raise AssertionError(f"Unknown dimensions {name}")
    dataset_dict = {
        "name": name,
        "kwargs": {
            "center": "center-" + center,
            "data_dir": os.path.join(data_dir, project_name),
            "crop_dir": os.path.join(crop_dir, project_name),
            "data_subsets": data_subsets,
            "type": type,
            "size": size,
            "transform": set_transforms,
        },
        "batch_size": batch_size,
        "virtual_batch_multiplier": virtual_batch_multiplier,
        "workers": workers,
    }
    print(
        "`{}_dataset_dict` dictionary successfully created with: \n -- {} images accessed from {}, "
        "\n -- number of images per epoch equal to {}, "
        "\n -- batch size set at {}, "
        "\n -- virtual batch multiplier set as {}.".format(
            type,
            [
                os.path.join(data_dir, project_name, subset, "images")
                for subset in data_subsets
            ],
            [
                os.path.join(crop_dir, project_name, subset, "images")
                for subset in data_subsets
            ],
            size,
            batch_size,
            virtual_batch_multiplier,
        )
    )
    return dataset_dict


def create_model_dict(input_channels, n_classes, name="2d"):
    """
    Creates `model_dict` dictionary from parameters.
    Parameters
    ----------
    input_channels: int
        1 indicates gray-channle image, 3 indicates RGB image.
    num_classes: list
        [4, 1] -> 4 indicates offset in x, offset in y, margin in x, margin in y; 1 indicates seediness score
    name: string
    """
    model_dict = {
        "name": "TrackERFNet"
        if name == "2d"
        else AssertionError(f"Unknown dimensions {name}"),
        "kwargs": {
            "n_classes": n_classes,
            "input_channels": input_channels,
        },
    }
    print(
        "`model_dict` dictionary successfully created with: \n -- num of classes equal to {}, \n -- input channels equal to {}, \n -- name equal to {}".format(
            n_classes, input_channels, name
        )
    )
    return model_dict


def create_loss_dict(foreground_weight=1, n_sigma=2, w_inst=1, w_var=10, w_seed=1):
    """
    Creates `loss_dict` dictionary from parameters.
    Parameters
    ----------
    foreground_weight: int
    w_inst: int/float
        weight on IOU loss
    w_var: int/float
        weight on variance loss
    w_seed: int/float
        weight on seediness loss
    """
    loss_dict = {
        "lossOpts": {
            "n_sigma": n_sigma,
            "foreground_weight": foreground_weight,
        },
        "lossW": {
            "w_inst": w_inst,
            "w_var": w_var,
            "w_seed": w_seed,
        },
    }
    print(
        "`loss_dict` dictionary successfully created with: \n -- foreground weight equal to {:.3f}, \n -- w_inst equal to {}, \n -- w_var equal to {}, \n -- w_seed equal to {}".format(
            foreground_weight, w_inst, w_var, w_seed
        )
    )
    return loss_dict


def create_configs(
    save_dir,
    resume_path,
    display=False,
    display_it=5,
    n_epochs=200,
    train_lr=5e-4,
    cuda=True,
    save=True,
    grid_y=1024,
    grid_x=1024,
    pixel_y=1,
    pixel_x=1,
):
    """
    Creates `configs` dictionary from parameters.
    Parameters
    ----------
    save_dir: str
        Path to where the experiment is saved
    resume_path: str
        Path to where the trained model (for example, checkpoint.pth) lives
    display: boolean
        If 'True', then realtime display of images, ground-truth, predictions are shown
    display_it: int
        Shows display every n training/val steps (display_it = n)
    n_epochs: int
        Total number of epochs
    train_lr: float
        Starting learning rate
    cuda: boolean
        If True, use GPU
    save: boolean
        If True, then results are saved
    n_y: int
        Size in y dimension of the largest evaluation image
    n_x: int
        Size in x dimension of the largest evaluation image
    pixel_y: int
        Pixel size in y
    pixel_x: int
        Pixel size in x
    """

    configs = dict(
        train_lr=train_lr,
        n_epochs=n_epochs,
        cuda=cuda,
        display=display,
        display_it=display_it,
        save=save,
        save_dir=save_dir,
        resume_path=resume_path,
        grid_y=grid_y,
        grid_x=grid_x,
        pixel_y=pixel_y,
        pixel_x=pixel_x,
    )
    print(
        "`configs` dictionary successfully created with: "
        "\n -- n_epochs equal to {}, "
        "\n -- display equal to {}, "
        "\n -- save_dir equal to {}, "
        "\n -- n_y equal to {}, "
        "\n -- n_x equal to {}, ".format(n_epochs, display, save_dir, grid_y, grid_x)
    )
    return configs
