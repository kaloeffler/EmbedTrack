"""
Original work Copyright 2019 Davy Neven,  KU Leuven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))
Modified work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
Modifications: remove 3d parts; change lr scheduler; change visualization; train/ eval on image pairs
"""
import shutil

import torch
import os


from embedtrack.criterions.loss import EmbedTrackLoss
from embedtrack.utils.utils import get_indices_pandas
from embedtrack.models.net import TrackERFNet
from embedtrack.datasets.dataset import get_dataset
from embedtrack.utils.logging import (
    AverageMeter,
    Logger,
)
from embedtrack.utils.clustering import Cluster
from embedtrack.utils.visualize import VisualizeTraining
from tqdm import tqdm


torch.backends.cudnn.benchmark = True
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
def train(virtual_batch_multiplier, n_sigma, args):
    # define meters
    loss_meter = AverageMeter()
    loss_parts_meter = {
        key: AverageMeter() for key in ["instance", "variance", "seed", "track"]
    }

    # put model into training mode
    model.train()
    for param_group in optimizer.param_groups:
        print("learning rate: {}".format(param_group["lr"]))

    optimizer.zero_grad()  # Reset gradients tensors

    for i, sample in enumerate(tqdm(train_dataset_it)):
        curr_frames = sample["image_curr"]  # curr frames
        prev_frames = sample["image_prev"]  # prev frames
        offset = sample["flow"].squeeze(1).to(device)  # 1YX
        seg_curr, seg_prev, tracking = model(curr_frames, prev_frames)  # B 5 Y X
        output = (torch.cat([seg_curr, seg_prev], dim=0), tracking)
        instances = torch.cat(
            [sample["instance_curr"], sample["instance_prev"]], dim=0
        ).squeeze(1)
        class_labels = torch.cat(
            [sample["label_curr"], sample["label_prev"]], dim=0
        ).squeeze(1)
        center_images = torch.cat(
            [sample["center_image_curr"], sample["center_image_prev"]], dim=0
        ).squeeze(1)
        loss, loss_parts = loss_fcn(
            output, instances, class_labels, center_images, offset, **args
        )
        loss = loss / virtual_batch_multiplier  # Normalize our loss (if averaged)
        loss = loss.mean()
        loss.backward()  # Backward pass

        if (i + 1) % virtual_batch_multiplier == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            scheduler.step()

            optimizer.zero_grad()  # Reset gradients tensors
        loss_meter.update(loss.item())
        for key in loss_parts_meter.keys():
            loss_parts_meter[key].update(loss_parts[key].mean().item())
    loss_part_avg = {
        key: meter.avg * virtual_batch_multiplier
        for key, meter in loss_parts_meter.items()
    }
    return loss_meter.avg * virtual_batch_multiplier, loss_part_avg


def train_vanilla(
    display,
    display_it,
    grid_x,
    grid_y,
    pixel_x,
    pixel_y,
    n_sigma,
    args,
):  # this is without virtual batches!

    # define meters
    loss_meter = AverageMeter()
    loss_parts_meter = {
        key: AverageMeter() for key in ["instance", "variance", "seed", "track"]
    }

    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print("learning rate: {}".format(param_group["lr"]))
    for i, sample in enumerate(tqdm(train_dataset_it)):

        curr_frames = sample["image_curr"]  # curr frames
        prev_frames = sample["image_prev"]  # prev frames
        offset = sample["flow"].squeeze(1).to(device)  # 1YX
        seg_curr, seg_prev, tracking = model(
            curr_frames, prev_frames
        )  # B 5 Y X, B 5 Y X, B 2 Y X
        output = (torch.cat([seg_curr, seg_prev], dim=0), tracking)
        instances = torch.cat(
            [sample["instance_curr"], sample["instance_prev"]], dim=0
        ).squeeze(1)
        class_labels = torch.cat(
            [sample["label_curr"], sample["label_prev"]], dim=0
        ).squeeze(1)
        center_images = torch.cat(
            [sample["center_image_curr"], sample["center_image_prev"]], dim=0
        ).squeeze(1)
        loss, loss_parts = loss_fcn(
            output, instances, class_labels, center_images, offset, **args
        )
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item())
        for key in loss_parts_meter.keys():
            loss_parts_meter[key].update(loss_parts[key].mean().item())
        # visualize data
        if display_it is not None:
            if i % display_it == 0 and i != 0:
                prediction = (output[0][0], output[1][0])
                ground_truth = (
                    instances[0].to(device),
                    center_images[0].to(device),
                    offset[0].to(device),
                )
                prev_instance = instances[len(instances) // 2].to(device)
                image_pair = (sample["image_curr"][0], sample["image_prev"][0])
                visualize_training(prediction, ground_truth, prev_instance, image_pair)

    loss_part_avg = {key: meter.avg for key, meter in loss_parts_meter.items()}
    return loss_meter.avg, loss_part_avg


def val(virtual_batch_multiplier, n_sigma, calc_iou, args):
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    loss_parts_meter = {
        key: AverageMeter() for key in ["instance", "variance", "seed", "track"]
    }
    # put model into eval mode
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            curr_frames = sample["image_curr"]  # curr frames
            prev_frames = sample["image_prev"]  # prev frames
            offset = sample["flow"].squeeze(1).to(device)  # 1YX
            seg_curr, seg_prev, tracking = model(
                curr_frames, prev_frames
            )  # B 5 Y X, B 5 Y X, B 2 Y X
            output = (torch.cat([seg_curr, seg_prev], dim=0), tracking)
            instances = torch.cat(
                [sample["instance_curr"], sample["instance_prev"]], dim=0
            ).squeeze(1)
            class_labels = torch.cat(
                [sample["label_curr"], sample["label_prev"]], dim=0
            ).squeeze(1)
            center_images = torch.cat(
                [sample["center_image_curr"], sample["center_image_prev"]], dim=0
            ).squeeze(1)
            loss, loss_parts = loss_fcn(
                output,
                instances,
                class_labels,
                center_images,
                offset,
                **args,
                iou=calc_iou,
                iou_meter=iou_meter,
            )
            loss = loss.mean()
            loss = loss / virtual_batch_multiplier
            loss_meter.update(loss.item())
            for key in loss_parts_meter.keys():
                loss_parts_meter[key].update(loss_parts[key].mean().item())
    loss_part_avg = {key: meter.avg for key, meter in loss_parts_meter.items()}
    return loss_meter.avg * virtual_batch_multiplier, iou_meter.avg, loss_part_avg


def val_vanilla(
    display,
    display_it,
    grid_x,
    grid_y,
    pixel_x,
    pixel_y,
    n_sigma,
    calc_iou,
    args,
):
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    loss_parts_meter = {
        key: AverageMeter() for key in ["instance", "variance", "seed", "track"]
    }

    # put model into eval mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            curr_frames = sample["image_curr"]  # curr frames
            prev_frames = sample["image_prev"]  # prev frames
            offset = sample["flow"].squeeze(1).to(device)  # 1YX
            seg_curr, seg_prev, tracking = model(
                curr_frames, prev_frames
            )  # B 5 Y X, B 5 Y X, B 2 Y X
            output = (torch.cat([seg_curr, seg_prev], dim=0), tracking)
            instances = torch.cat(
                [sample["instance_curr"], sample["instance_prev"]], dim=0
            ).squeeze(1)
            class_labels = torch.cat(
                [sample["label_curr"], sample["label_prev"]], dim=0
            ).squeeze(1)
            center_images = torch.cat(
                [sample["center_image_curr"], sample["center_image_prev"]], dim=0
            ).squeeze(1)
            loss, loss_parts = loss_fcn(
                output,
                instances,
                class_labels,
                center_images,
                offset,
                **args,
                iou=calc_iou,
                iou_meter=iou_meter,
            )

            loss = loss.mean()

            loss_meter.update(loss.item())
            for key in loss_parts_meter.keys():
                loss_parts_meter[key].update(loss_parts[key].mean().item())
    loss_part_avg = {key: meter.avg for key, meter in loss_parts_meter.items()}
    return loss_meter.avg, iou_meter.avg, loss_part_avg


def save_checkpoint(state, is_best, epoch, save_dir, name="checkpoint.pth"):
    print("=> saving checkpoint")
    file_name = os.path.join(save_dir, name)
    torch.save(state, file_name)
    if epoch % 10 == 0:
        file_name2 = os.path.join(save_dir, str(epoch) + "_" + name)
        torch.save(state, file_name2)
    if is_best:
        shutil.copyfile(file_name, os.path.join(save_dir, "best_iou_model.pth"))


def begin_training(
    train_dataset_dict,
    val_dataset_dict,
    model_dict,
    loss_dict,
    configs,
):

    if configs["save"]:
        if not os.path.exists(configs["save_dir"]):
            os.makedirs(configs["save_dir"])

    if configs["display"]:
        plt.ion()
    else:
        plt.ioff()
        plt.switch_backend("agg")

    # define global variables
    global train_dataset_it, val_dataset_it, model, loss_fcn, optimizer, cluster, visualize_training, scheduler

    # train dataloader

    train_dataset = get_dataset(
        train_dataset_dict["name"], train_dataset_dict["kwargs"]
    )
    train_dataset_it = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_dataset_dict["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=train_dataset_dict["workers"],
        pin_memory=True if configs["cuda"] else False,
    )

    # val dataloader
    val_dataset = get_dataset(val_dataset_dict["name"], val_dataset_dict["kwargs"])
    val_dataset_it = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_dataset_dict["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=val_dataset_dict["workers"],
        pin_memory=True if device == "cuda" else False,
    )

    # set model
    model = TrackERFNet(**model_dict["kwargs"])
    model.init_output(loss_dict["lossOpts"]["n_sigma"])
    model = torch.nn.DataParallel(model).to(device)

    cluster = Cluster(
        configs["grid_y"],
        configs["grid_x"],
        configs["pixel_y"],
        configs["pixel_x"],
    )

    cluster = cluster.to(device)
    loss = EmbedTrackLoss(
        grid_y=configs["grid_y"],
        grid_x=configs["grid_x"],
        pixel_y=configs["pixel_y"],
        pixel_x=configs["pixel_x"],
        cluster=cluster,
        **loss_dict["lossOpts"],
    )

    loss_fcn = torch.nn.DataParallel(loss).to(device)

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["train_lr"], weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=configs["train_lr"],
        total_steps=configs["n_epochs"]
        * len(train_dataset_it)
        // train_dataset_dict["virtual_batch_multiplier"],
    )

    visualize_training = VisualizeTraining(
        cluster,
        configs["save_dir"],
        grid_x=configs["grid_x"],
        grid_y=configs["grid_y"],
        pixel_x=configs["pixel_x"],
        pixel_y=configs["pixel_y"],
        n_sigma=loss_dict["lossOpts"]["n_sigma"],
    )

    # Logger
    logger = Logger(("train", "val", "iou"), "loss")

    # resume
    start_epoch = 0
    best_iou = 0
    if configs["resume_path"] is not None and os.path.exists(configs["resume_path"]):
        print("Resuming model from {}".format(configs["resume_path"]))
        state = torch.load(configs["resume_path"])
        start_epoch = state["epoch"] + 1
        best_iou = state["best_iou"]
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])
        logger.data = state["logger_data"]

    for epoch in range(start_epoch, configs["n_epochs"]):

        print("Starting epoch {}".format(epoch))
        if train_dataset_dict["virtual_batch_multiplier"] > 1:
            train_loss, train_loss_parts = train(
                virtual_batch_multiplier=train_dataset_dict["virtual_batch_multiplier"],
                n_sigma=loss_dict["lossOpts"]["n_sigma"],
                args=loss_dict["lossW"],
            )
        else:
            train_loss, train_loss_parts = train_vanilla(
                display=configs["display"],
                display_it=configs["display_it"],
                n_sigma=loss_dict["lossOpts"]["n_sigma"],
                grid_x=configs["grid_x"],
                grid_y=configs["grid_y"],
                pixel_x=configs["pixel_x"],
                pixel_y=configs["pixel_y"],
                args=loss_dict["lossW"],
            )

        if val_dataset_dict["virtual_batch_multiplier"] > 1:
            if epoch > configs["n_epochs"] / 4:
                calc_iou = True
            else:
                calc_iou = False
            val_loss, val_iou, val_loss_parts = val(
                virtual_batch_multiplier=val_dataset_dict["virtual_batch_multiplier"],
                calc_iou=calc_iou,
                n_sigma=loss_dict["lossOpts"]["n_sigma"],
                args=loss_dict["lossW"],
            )
        else:
            if epoch > configs["n_epochs"] / 4:
                calc_iou = True
            else:
                calc_iou = False
            val_loss, val_iou, val_loss_parts = val_vanilla(
                display=configs["display"],
                display_it=configs["display_it"],
                n_sigma=loss_dict["lossOpts"]["n_sigma"],
                grid_x=configs["grid_x"],
                grid_y=configs["grid_y"],
                pixel_x=configs["pixel_x"],
                pixel_y=configs["pixel_y"],
                calc_iou=calc_iou,
                args=loss_dict["lossW"],
            )

        print("===> train loss: {:.6f}".format(train_loss))
        print("===> val loss: {:.4f}, val iou: {:.4f}".format(val_loss, val_iou))

        logger.add("train", train_loss)
        logger.add("val", val_loss)
        logger.add("iou", val_iou)
        for key, value in train_loss_parts.items():
            name = "train_" + key
            if name not in logger.data:
                logger.data[name] = []
            logger.add(name, train_loss_parts[key])

        for key, value in val_loss_parts.items():
            name = "val_" + key
            if name not in logger.data:
                logger.data[name] = []
            logger.add(name, val_loss_parts[key])

        logger.plot(save=configs["save"], save_dir=configs["save_dir"])

        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        if configs["save"]:
            state = {
                "epoch": epoch,
                "best_iou": best_iou,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "logger_data": logger.data,
            }
        save_checkpoint(state, is_best, epoch, save_dir=configs["save_dir"])
