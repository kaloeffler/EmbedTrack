"""
Original work Copyright 2019 Davy Neven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))
Modified work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler (MIT License)
Modifications: changed IOU calculation; extended with tracking loss
"""

import torch
import torch.nn as nn
from embedtrack.criterions.lovasz_losses import lovasz_hinge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbedTrackLoss(nn.Module):
    def __init__(
        self,
        cluster,
        grid_y=1024,
        grid_x=1024,
        pixel_y=1,
        pixel_x=1,
        n_sigma=2,
        foreground_weight=1,
    ):
        """
        Loss for training the EmbedTrack net.
        Args:
            cluster (Callable): cluster the predictions to instances
            grid_x (int): size of the grid in x direction
            grid_y (int): size of the grid in y direction
            pixel_x (int): size of a pixel
            pixel_y (int): size of a pixel
            n_sigma (int): number of channels estimating sigma (which is used to estimate the object size)
            foreground_weight (int): weight of the foreground compare to the background
        """
        super().__init__()

        print(
            "Created spatial emb loss function with: n_sigma: {}, foreground_weight: {}".format(
                n_sigma, foreground_weight
            )
        )
        print("*************************")
        self.cluster = cluster
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        # coordinate map
        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
        yxm = torch.cat((ym, xm), 0)

        self.register_buffer("yxm", yxm)
        self.register_buffer("yx_shape", torch.tensor(self.yxm.size()[1:]).view(-1, 1))

    def forward(
        self,
        predictions,
        instances,
        labels,
        center_images,
        offsets,
        w_inst=1,
        w_var=10,
        w_seed=1,
        iou=False,
        iou_meter=None,
    ):
        """

        Args:
            predictions (tuple): tuple of torch tensors containing the network output
            instances (torch.Tensor): ground truth instance segmentation masks
            labels (torch.Tensor): semantic segmentation masks
            center_images (torch.Tensor): masks containing the gt cell center position
            offsets (torch.Tensor): masks containing the shift between cell centers of two successive frames
            w_inst (int): weight for the instance loss
            w_var (int): weight for the variance loss
            w_seed (int): weight for the seed loss
            iou (bool): if True, calculate the IOU of the instance segmentation
            iou_meter (Callable): contains the calculated IOU scores

        Returns: (torch.Tensor) loss, (dict) values of the different loss parts

        """
        segmentation_predictions, offset_predictions = predictions
        # instances B 1 Y X
        batch_size, height, width = (
            segmentation_predictions.size(0),
            segmentation_predictions.size(2),
            segmentation_predictions.size(3),
        )

        yxm_s = self.yxm[:, 0:height, 0:width]  # N x h x w if 2D images: N=2

        loss = torch.tensor(0, device=device, dtype=torch.float)
        track_loss = 0
        track_count = 0

        loss_values = {
            "instance": 0,
            "variance": 0,
            "seed": 0,
            "track": 0,
        }
        for b in range(0, batch_size):
            seed_loss_it = 0
            seed_loss_count = 0
            segm_offsets = torch.tanh(segmentation_predictions[b, 0:2])
            spatial_emb = segm_offsets + yxm_s
            if b < batch_size // 2:
                track_offsets = torch.tanh(offset_predictions[b, ...])
                tracking_emb = yxm_s - track_offsets
            # edited to be between 0...1 -> scaling with exp(K*x)
            sigma = torch.sigmoid(
                segmentation_predictions[b, 2 : 2 + self.n_sigma]
            )  # n_sigma x h x w

            seed_map = torch.sigmoid(
                segmentation_predictions[b, 2 + self.n_sigma : 2 + self.n_sigma + 1]
            )  # 1 x h x w
            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0

            instance = instances[b].unsqueeze(0)  # 1 x h x w
            label = labels[b].unsqueeze(0)  # 1 x h x w
            center_image = center_images[b].unsqueeze(0)  # 1 x h x w

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # regress bg to zero
            bg_mask = label == 0

            if bg_mask.sum() > 0:
                seed_loss_it += torch.mean(torch.pow(seed_map[bg_mask] - 0, 2))
                seed_loss_count += bg_mask.sum()
            if len(instance_ids) == 0:  # background image
                continue
            # n_dim; n_sigma
            all_sigmas = torch.stack(
                [
                    sigma[:, instance.eq(inst_id).squeeze()].mean(dim=1)
                    for inst_id in instance_ids
                ]
            ).T

            for i, inst_id in enumerate(instance_ids):
                in_mask = instance.eq(inst_id)  # 1 x h x w
                center_mask = in_mask & center_image.byte().bool()
                if center_mask.sum().eq(0):
                    continue
                if center_mask.sum().eq(1):
                    center = yxm_s[center_mask.expand_as(yxm_s)].view(2, 1, 1)
                else:
                    xy_in = yxm_s[in_mask.expand_as(yxm_s)].view(
                        2, -1
                    )  # TODO --> should this edge case change!
                    center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(sigma)].view(self.n_sigma, -1)

                s = all_sigmas[:, i].view(self.n_sigma, 1, 1)  # n_sigma x 1 x 1

                # calculate var loss before exp
                var_loss = var_loss + torch.mean(torch.pow(sigma_in - s.detach(), 2))

                s = torch.exp(
                    s * 10
                )  # if sigmoid constrained 0...1 before exp afterwards scale 1...22026 - more than enough range to simulate pix size objects and large objects!
                dist = torch.exp(
                    -1
                    * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True)
                )

                # apply lovasz-hinge loss
                instance_loss = instance_loss + lovasz_hinge(
                    dist * 2 - 1, in_mask.to(device)
                )

                # seed loss
                seed_loss_it += self.foreground_weight * torch.mean(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2)
                )
                seed_loss_count += in_mask.sum()

                # segmentation branch predictions where concatinated (frames t, frames t-1)
                # since tracking offset is calculated between t->t-1
                # the tracking batch has only half the length compared to the segmentation predictions
                if b < (batch_size // 2):
                    index_gt_center = (in_mask & center_image.byte().bool()).squeeze()
                    if index_gt_center.sum().eq(0):
                        continue
                    # this is -offset due to how they were calculated before
                    gt_prev_center_yxms = (
                        (
                            yxm_s[:, index_gt_center]
                            - offsets[b, :, index_gt_center] / self.yx_shape
                        )
                        .view(-1, 1, 1)
                        .float()
                    )
                    dist_tracking = torch.exp(
                        -1
                        * torch.sum(
                            torch.pow(tracking_emb - gt_prev_center_yxms, 2) * s,
                            0,
                            keepdim=True,
                        )
                    )
                    track_loss = track_loss + lovasz_hinge(
                        dist_tracking * 2 - 1, in_mask.to(device)
                    )
                    track_count += 1

            seed_loss += seed_loss_it

            # calculate instance IOU
            if iou:
                instance_pred = self.cluster.cluster_pixels(
                    segmentation_predictions[b], n_sigma=2, return_on_cpu=False
                )
                iou_scores = calc_iou_full_image(
                    instances[b].detach(),
                    instance_pred.detach(),
                )
                for score in iou_scores:
                    iou_meter.update(score)
            # seed_loss = seed_loss / torch.prod(torch.tensor(instances.shape[1:]))
            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss
            loss_values["instance"] += (
                w_inst * instance_loss.detach()
                if isinstance(instance_loss, torch.Tensor)
                else torch.tensor(w_inst * instance_loss).float().to(device)
            )
            loss_values["variance"] += (
                w_var * var_loss.detach()
                if isinstance(var_loss, torch.Tensor)
                else torch.tensor(w_var * var_loss).float().to(device)
            )
            loss_values["seed"] += (
                w_seed * seed_loss.detach()
                if isinstance(seed_loss, torch.Tensor)
                else torch.tensor(w_seed * seed_loss).float().to(device)
            )
            loss_values["track"] += (
                track_loss.detach()
                if isinstance(track_loss, torch.Tensor)
                else torch.tensor(track_loss).float().to(device)
            )

        if track_count > 0:
            track_loss /= track_count
        loss += track_loss
        loss = loss / (b + 1)

        return loss + segmentation_predictions.sum() * 0, loss_values


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou


def calc_iou_full_image(gt, prediction):
    """Calculate all IOUs in the image crop"""
    gt_labels = torch.unique(gt)
    gt_labels = gt_labels[gt_labels > 0]
    pred_labels = prediction[prediction > 0].unique()
    # go through gt labels
    ious = []
    matched_pred_labels = []
    for gt_l in gt_labels:
        gt_mask = gt.eq(gt_l)
        overlapping_pred_labels = prediction[gt_mask].unique()
        overlapping_pred_labels = overlapping_pred_labels[overlapping_pred_labels > 0]
        if not len(overlapping_pred_labels):  # false negative
            ious.append(0)
            continue
        # otherwise assign to gt mask the prediction with largest iou
        # calculate_iou returns single float which is on the cpu
        gt_ious = torch.tensor(
            [
                calculate_iou(gt_mask, prediction.eq(p_l))
                for p_l in overlapping_pred_labels
            ]
        )
        if len(gt_ious) > 0:
            idx_max_iou = torch.argmax(gt_ious)
            ious.append(gt_ious[idx_max_iou])
            matched_pred_labels.append(overlapping_pred_labels[idx_max_iou])

    # add not matched pred labels by adding iou==0 (FPs)
    if len(matched_pred_labels) > 0:
        matched_pred_labels = torch.stack(matched_pred_labels)
        # (pred_labels[..., None] == matched_pred_labels).any(-1) equvalent to np.isin(pred_labels, matched_pred_labels)
        num_non_matched = (
            ~(pred_labels[..., None] == matched_pred_labels).any(-1)
        ).sum()
    else:
        num_non_matched = len(pred_labels)
    ious.extend([0] * num_non_matched)
    return ious
