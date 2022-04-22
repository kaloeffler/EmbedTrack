"""
Original work Copyright 2019 Davy Neven,  KU Leuven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))
Modified work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
Modifications: remove cluster_with_gt; cluster method (cluster) modified (nor: cluster_pixels); add smoothing_filt
added find_seeds
"""
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Cluster(nn.Module):
    def __init__(self, grid_y, grid_x, pixel_y, pixel_x):
        super().__init__()
        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
        yxm = torch.cat((ym, xm), 0)

        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.register_buffer("yxm", yxm)
        smoothing_filt = torch.nn.Conv2d(
            1, 1, (3, 3), padding="same", padding_mode="reflect"
        )
        smoothing_filt.weight.data = torch.ones(1, 1, 3, 3) / 9
        smoothing_filt.to(device)
        self.smoothing_filt = smoothing_filt

    def cluster_pixels(
        self, prediction, n_sigma=2, min_obj_size=10, return_on_cpu=True
    ):
        """
        Cluster segmentation prediction into cell instances.
        Args:
            prediction: torch.tensor
                segmentation prediction of the model
            n_sigma: int
                number of channels to estimate sigma
            min_obj_size: float
                minimum object size
            return_on_cpu: bool
                return the instance map on cpu or gpu

        Returns: torch.tensor (instance segmentation map)

        """

        height, width = prediction.size(1), prediction.size(2)
        yxm_s = self.yxm[:, 0:height, 0:width]

        sigma = torch.exp(
            torch.sigmoid(prediction[2 : 2 + n_sigma]) * 10
        )  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma : 2 + n_sigma + 1])  # 1 x h x w

        spatial_emb = torch.tanh(prediction[0:2]) + yxm_s
        with torch.no_grad():
            # 2 x h x w
            sigma = self.smoothing_filt(sigma[:, np.newaxis, ...]).squeeze()

        mask = (seed_map > 0.5).squeeze()

        selected_pixels = (
            seed_map > 0.5
        )  # use only pixels with very high seediness scores as potential seeds!
        seeds = self.find_seeds(
            spatial_emb[selected_pixels.expand_as(spatial_emb)].view(2, -1),
            self.smoothing_filt.weight.numel() * 0.5,
        ).expand_as(seed_map)
        # filter out seed pixels clustered to the background
        seeds = seeds & (seed_map > 0.1)
        seed_map_masked = seed_map[seeds]
        spatial_emb_masked = spatial_emb[seeds.expand_as(spatial_emb)].view(2, -1)
        sigma_masked = sigma[seeds.expand_as(sigma)].view(2, -1)

        instance_map_masked = torch.zeros(
            height, width, device=device, dtype=torch.short
        )
        dist_map = torch.zeros(height, width, dtype=prediction.dtype, device=device)
        unclustered = torch.ones(seeds.sum(), device=device).short()

        count = 1
        while unclustered.sum():
            seed_index = (seed_map_masked * unclustered.float()).argmax().item()
            center = spatial_emb_masked[:, seed_index].reshape(-1, 1, 1)
            s = sigma_masked[:, seed_index].reshape(-1, 1, 1)
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            # 0.5 due to training lovas loss exp(-|p_i-c_j|**2*s_j) > 0.5
            proposal = (dist > dist_map) & (dist > 0.5) & mask
            colliding_pixel = (instance_map_masked > 0)[dist > 0.5].sum()
            if ((colliding_pixel / proposal.sum()) < 0.5) and (
                proposal.sum() >= min_obj_size
            ):
                instance_map_masked[proposal] = count
                count += 1
                unclustered[proposal[seeds.squeeze()]] = 0

            dist_map[proposal] = dist[proposal]
            unclustered[seed_index] = 0
        if not return_on_cpu:
            return instance_map_masked

        return instance_map_masked.cpu()

    def find_seeds(self, spatial_emb_masked, min_cluster_points):
        """
        Find potential cell centers.
        Args:
            spatial_emb_masked: torch.tensor
                estimated positions of cell centers
            min_cluster_points: int
                min number of pixels clustered in the neighborhood

        Returns: torch.tensor
            map of potential cell centers

        """
        # select pix with reliable estimation
        pix_positions = degrid(
            spatial_emb_masked,
            torch.tensor([self.grid_y, self.grid_x]).reshape(-1, 1).to(device),
            torch.tensor([self.pixel_y, self.pixel_x]).reshape(-1, 1).to(device),
        )
        pix_positions = pix_positions[:, ~torch.any(pix_positions < 0, dim=0)]
        # because yxm has shape 2, y,x in 2d -> compare pix positions and yxm.shape shifted by 1
        pix_positions = pix_positions[:, ~(pix_positions[0] > self.yxm.shape[1] - 1)]
        pix_positions = pix_positions[:, ~(pix_positions[1] > self.yxm.shape[2] - 1)]
        clustered_pixels = torch.zeros(self.yxm.shape[1:], device=device)
        pix_positions_unique, counts = pix_positions.unique(return_counts=True, dim=1)
        if len(pix_positions_unique) > 0:
            clustered_pixels[
                pix_positions_unique[0].long(), pix_positions_unique[1].long()
            ] = counts.float()
        with torch.no_grad():
            # multiply with filter size to get a sum filter and not an avg
            clustered_pixels_smoothed = (
                self.smoothing_filt(
                    clustered_pixels[np.newaxis, np.newaxis, ...]
                ).squeeze()
                * self.smoothing_filt.weight.data.numel()
            )
        seeds = clustered_pixels_smoothed >= min_cluster_points
        return seeds


def degrid(meter, grid_size, pixel_size):
    return torch.round(meter * (grid_size - 1) / pixel_size).int()  # + 1)
