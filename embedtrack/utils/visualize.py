"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License

mypause function from work of Davy Neven (2019),  KU Leuven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))

"""
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import DataFrame


class VisualizeTraining:
    def __init__(
        self,
        cluster,
        export_path,
        export_data=True,
        export_format=".pdf",
        grid_x=1024,
        grid_y=1024,
        pixel_x=1,
        pixel_y=1,
        n_sigma=2,
        color_map="plasma",
    ):
        """
        Visualize the data during training.
        Args:
            cluster (Callable): clustering instance to cluster the predictions into instances
            export_path (string): path to where store the visualizations
            export_data (boolean): if True store visualisations in export_path
            export_format (string): format to store the visualization figures (png, pdf,...)
            grid_x (int): size of the grid in x direction
            grid_y (int): size of the grid in y direction
            pixel_x (int): size of a pixel
            pixel_y (int): size of a pixel
            n_sigma (int): dimensionality of sigma (in 2d: 2)
            color_map (string): how to color the predictions in the plots
        """
        self.subplt_size = (5, 5)
        n_plots_segm = (1, 5)
        n_plots_track = (2, 3)
        n_plots_offset = (2, 3)
        n_plots_raw_img = (2, 1)
        self.cbar_size = 5  # in percent
        self.figures = dict()
        self.axes = dict()
        self.add_plot(n_plots_segm, "segmentation")
        self.add_plot(n_plots_track, "tracking")
        self.add_plot(n_plots_offset, "offsets")
        self.add_plot(n_plots_raw_img, "raw_img_pair")

        self.patches = []

        self.grid = calc_grid(grid_x, grid_y, pixel_x, pixel_y).cuda()
        self.grid_shape = torch.tensor(self.grid.size()[1:]).view(-1, 1).cuda()
        self.n_sigma = n_sigma
        self.cluster = cluster

        self.export_path = export_path
        self.export_data = export_data
        self.export_format = export_format

        self.color_bars = dict()

        self.gray_cmp = ListedColormap(sns.color_palette(color_map, 256))
        self.gray_cmp.set_bad("k")
        self.gray_cmp.set_under("gray")

        custom_col = [
            self.gray_cmp.colors[len(self.gray_cmp.colors) // 2],
            self.gray_cmp.colors[0],
            mpl.colors.CSS4_COLORS["white"],
            self.gray_cmp.colors[-1],
            self.gray_cmp.colors[len(self.gray_cmp.colors) // 2],
        ]

        self.cyclic_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "cyclic", custom_col, N=256, gamma=1
        )
        self.cmap_pix_pred = mpl.colors.LinearSegmentedColormap.from_list(
            "cmap_class",
            ["black", "white", "blue", "orange"],
            N=4,
            gamma=1,
        )
        self.norm_pix_pred = mpl.colors.BoundaryNorm(
            np.linspace(0, 3, 4), self.cmap_pix_pred.N
        )
        self.cyclic_cmap.set_bad(color="k")
        self.cyclic_cmap.set_under("gray")

        # will be set to false after first call of rescale
        self.rescale = True

    def set_ax_off(self, ax):
        if isinstance(ax, np.object):
            for sub_ax in ax.flatten():
                sub_ax.axis("off")
        else:
            ax.axis("off")

    def calc_fig_size(self, n_subplots, fig_size):
        width = n_subplots[0] * fig_size[0]
        height = n_subplots[1] * fig_size[1]
        return (height, width)

    def __call__(self, prediction, ground_truth, prev_instance_mask, image_pair):
        """Generate all visualization plots."""
        self.vis_segmentation(prediction, ground_truth, prev_instance_mask)
        self.vis_tracking(prediction, ground_truth, prev_instance_mask)
        self.vis_loss_sigma_and_offsets(prediction, ground_truth)
        self.vis_raw_img_pair(image_pair)
        if self.export_data:
            self.save_figures()

    def add_plot(self, n_plots, fig_name):
        """Add another visualization plot."""
        fig, ax = plt.subplots(
            *n_plots, figsize=self.calc_fig_size(n_plots, self.subplt_size)
        )
        self.figures[fig_name] = fig
        self.axes[fig_name] = ax
        self.set_ax_off(ax)

    def resize_plots(self):
        """Resize plots so all subplots are approximately of the same size when saved."""
        # get largest bbox -> resize all figures accordingly
        largest_plot = 0
        for fig_name, ax in self.axes.items():
            self.figures[fig_name].tight_layout(h_pad=2, w_pad=2)
            largest_plot = max(largest_plot, ax.flatten()[0].bbox.width)
        for fig_name, fig in self.figures.items():
            ax = self.axes[fig_name].flatten()[0]
            ax_size = ax.bbox.width
            grid_shape = np.array([ax.get_gridspec().nrows, ax.get_gridspec().ncols])
            size_diff = largest_plot - ax_size
            # h, w
            diff_inches = size_diff / matplotlib.rcParams["figure.dpi"] * grid_shape
            # w, h
            new_fig_size = fig.get_size_inches() + diff_inches[::-1]
            fig.set_size_inches(w=new_fig_size[0], h=new_fig_size[1])
            fig.tight_layout(h_pad=2, w_pad=2)

    def save_figures(self):
        """Export plots to the specified format and save on disk."""
        if self.rescale:
            self.resize_plots()
            self.rescale = False

        time_stamp = datetime.now().strftime("%d-%m-%y--%H-%M-%S")
        for fig_name, figure in self.figures.items():
            file_name = fig_name + "_" + time_stamp + self.export_format
            full_file_path = os.path.join(
                self.export_path, "visualizations", fig_name, file_name
            )
            if not os.path.exists(Path(full_file_path).parent):
                os.makedirs(Path(full_file_path).parent)
            figure.savefig(
                full_file_path, dpi=400, bbox_inches="tight", pad_inches=0.05
            )

    def add_colorbar(self, fig, image, axis, key):
        """Add colorbar to a subplot."""
        if key not in self.color_bars:
            self.color_bars[key] = None
        # cax= cb.ax needed because color bars are created separately from sub figure
        # otherwise new color bars will be created in every time this func part is called
        if self.color_bars[key] is None:
            divider = make_axes_locatable(axis)
            cb = divider.append_axes("right", size=str(self.cbar_size) + "%", pad=0.05)
            self.color_bars[key] = cb
        else:
            self.color_bars[key].clear()
        fig.colorbar(image, ax=axis, cax=self.color_bars[key])
        # set log formatter of colorbar to linear formatter
        if isinstance(
            axis.images[-1].colorbar.formatter,
            matplotlib.ticker.LogFormatterSciNotation,
        ):
            edit_labels = log_to_scalar_ticks_label(
                self.color_bars[key].yaxis.get_ticklabels()
            )
            self.color_bars[key].yaxis._set_ticklabels(edit_labels)

    def vis_raw_img_pair(self, image_pair):
        img_t_1, img_t_0 = image_pair
        self.axes["raw_img_pair"][0].imshow(img_t_1.squeeze(), cmap="gray")
        self.axes["raw_img_pair"][0].set_title(r"Raw Image $t+1$")
        self.axes["raw_img_pair"][1].imshow(img_t_0.squeeze(), cmap="gray")
        self.axes["raw_img_pair"][1].set_title(r"Raw Image $t$")
        self.add_dummy_axis(self.axes["raw_img_pair"][0])
        self.add_dummy_axis(self.axes["raw_img_pair"][1])

    def vis_loss_sigma_and_offsets(self, prediction, ground_truth):
        """Visualizes the learned sigma maps and the error of the
        offsets predicted for segmentation and tracking."""
        with torch.no_grad():
            segm_prediction, offset_tracking_pred = prediction
            instance_mask, center_image, offset = ground_truth
            height, width = instance_mask.shape

            cell_mask = np.tile((instance_mask == 0).cpu()[np.newaxis, ...], (2, 1, 1))

            pos_tracking_gt = self.calc_gt_centroid_pos(
                instance_mask, center_image, offset
            )
            sigma = torch.exp(torch.sigmoid(segm_prediction[2 : 2 + self.n_sigma]) * 10)
            sigma[:, instance_mask == 0] = 0
            sigma_masked = np.ma.masked_array(sigma.cpu(), mask=cell_mask)
            c_ax = self.axes["offsets"][0][0].imshow(
                sigma_masked[0], cmap=self.gray_cmp
            )
            self.add_colorbar(
                self.figures["offsets"],
                c_ax,
                self.axes["offsets"][0][0],
                ("sigma_y", 0, 0),
            )
            c_ax = self.axes["offsets"][1][0].imshow(
                sigma_masked[1], cmap=self.gray_cmp
            )
            self.add_colorbar(
                self.figures["offsets"],
                c_ax,
                self.axes["offsets"][1][0],
                ("sigma_x", 1, 0),
            )

            pos_seg_pred = (
                torch.tanh(segm_prediction[0:2])
                + self.grid[:, 0:height, 0:width].contiguous()
            )
            pos_tracking_pred = self.grid[
                :, 0:height, 0:width
            ].contiguous() - torch.tanh(offset_tracking_pred)
            offset_segm_gt = self.calc_gt_centroid_pos(instance_mask, center_image)
            mask_missing = cell_ids_without_center(instance_mask, center_image).cpu()
            offset_segm_diff = torch.mul(
                offset_segm_gt - pos_seg_pred, self.grid_shape.reshape(-1, 1, 1)
            ).abs()
            offset_segm_diff[:, instance_mask == 0] = 0
            offset_segm_diff_masked = np.ma.masked_array(
                offset_segm_diff.cpu(), mask=cell_mask
            )
            offset_segm_diff_masked[:, mask_missing] = -10000
            c_ax = self.axes["offsets"][0][1].imshow(
                offset_segm_diff_masked[0],
                cmap=self.gray_cmp,
                norm=matplotlib.colors.SymLogNorm(
                    linthresh=self.grid_shape[0] / 10,
                    vmin=0,
                    vmax=self.grid_shape[0],
                    base=10,
                    linscale=3,
                ),
            )
            self.add_colorbar(
                self.figures["offsets"],
                c_ax,
                self.axes["offsets"][0][1],
                ("segm_offset_y", 0, 1),
            )
            c_ax = self.axes["offsets"][1][1].imshow(
                offset_segm_diff_masked[1],
                cmap=self.gray_cmp,
                norm=matplotlib.colors.SymLogNorm(
                    linthresh=self.grid_shape[1] / 10,
                    vmin=0,
                    vmax=self.grid_shape[1],
                    base=10,
                    linscale=3,
                ),
            )
            self.add_colorbar(
                self.figures["offsets"],
                c_ax,
                self.axes["offsets"][1][1],
                ("segm_offset_x", 0, 1),
            )

            pos_tracking_pred[:, instance_mask == 0] = 0
            pos_tracking_gt[:, instance_mask == 0] = 0
            offset_tracking_diff = torch.mul(
                pos_tracking_gt - pos_tracking_pred,
                self.grid_shape.reshape(-1, 1, 1),
            ).abs()
            offset_tracking_diff_masked = np.ma.masked_array(
                offset_tracking_diff.cpu(), mask=cell_mask
            )
            # mask cells with missing annotation with high neg value
            # which is used later by the cmap to mark them in another color
            offset_tracking_diff_masked[:, mask_missing] = -10000

            # scaling of the axis to show large offset errors but have
            # enough scale to show small errors as well
            c_ax = self.axes["offsets"][0][2].imshow(
                offset_tracking_diff_masked[0],
                cmap=self.gray_cmp,
                norm=matplotlib.colors.SymLogNorm(
                    linthresh=self.grid_shape[0] / 10,
                    vmin=0,
                    vmax=self.grid_shape[0],
                    base=10,
                    linscale=3,
                ),
            )
            self.add_colorbar(
                self.figures["offsets"],
                c_ax,
                self.axes["offsets"][0][2],
                ("track_offset_y", 0, 2),
            )
            c_ax = self.axes["offsets"][1][2].imshow(
                offset_tracking_diff_masked[1],
                cmap=self.gray_cmp,
                norm=matplotlib.colors.SymLogNorm(
                    linthresh=self.grid_shape[1] / 10,
                    vmin=0,
                    vmax=self.grid_shape[1],
                    base=10,
                    linscale=3,
                ),
            )
            self.add_colorbar(
                self.figures["offsets"],
                c_ax,
                self.axes["offsets"][1][2],
                ("track_offset_x", 1, 2),
            )

            self.axes["offsets"][0][0].set_title(r"Sigma y")
            self.axes["offsets"][1][0].set_title(r"Sigma x")
            self.axes["offsets"][0][1].set_title(
                r"Segm: Offset Error $|\mathbf{o}_y-\hat{\mathbf{o}}_y|$"
            )
            self.axes["offsets"][1][1].set_title(
                r"Segm: Offset Error $|\mathbf{o}_x-\hat{\mathbf{o}}_x|$"
            )
            self.axes["offsets"][0][2].set_title(
                r"Track: Offset Error $|\mathbf{o}_y-\hat{\mathbf{o}}_y|$"
            )
            self.axes["offsets"][1][2].set_title(
                r"Track: Offset Error $|\mathbf{o}_x-\hat{\mathbf{o}}_x|$"
            )

            plt.draw()
            self.mypause(0.0001)

    def calc_gt_flow(self, instance_mask, center_image, offset):
        """Calculate flow, the offset of each pixel belonging to
        a cell at t to its corresponding cell center at t-1."""
        height, width = instance_mask.shape
        yxm_s = self.grid[:, 0:height, 0:width].contiguous()  # 2 x h x w

        instance_ids = instance_mask.unique()
        instance_ids = instance_ids[instance_ids != 0]
        gt_flow_mask = torch.zeros_like(offset)
        for inst_id in instance_ids:
            in_mask = instance_mask.eq(inst_id)  # 1 x h x w
            index_gt_center = (in_mask & center_image.byte().bool()).squeeze()
            if index_gt_center.sum() > 0:
                gt_center_position = (
                    yxm_s[:, index_gt_center]
                    - offset[:, index_gt_center] / self.grid_shape
                )
            else:
                gt_center_position = yxm_s[:, in_mask.squeeze()]
            gt_flow_mask[:, in_mask.squeeze()] = (
                yxm_s[:, in_mask.squeeze()] - gt_center_position
            )
        return gt_flow_mask

    def vis_segmentation(self, prediction, ground_truth, instance_mask_prev_frame):
        """Visualize the segmentation prediction channels."""
        with torch.no_grad():
            segm_prediction, offset_tracking_pred = prediction
            instance_mask, center_image, offset = ground_truth
            height, width = instance_mask.shape

            pos_seg_pred = (
                torch.tanh(segm_prediction[0:2])
                + self.grid[:, 0:height, 0:width].contiguous()
            )
            idx_seg_pred = torch.stack(
                (
                    degrid(pos_seg_pred[0], self.cluster.grid_y, self.cluster.pixel_y),
                    degrid(pos_seg_pred[1], self.cluster.grid_x, self.cluster.pixel_x),
                )
            )
            idx_seg_pred = idx_seg_pred.reshape(2, -1)

            grid = torch.stack(
                (
                    torch.arange(0, instance_mask.shape[0])
                    .view(1, -1, 1)
                    .expand(1, *instance_mask.shape),
                    torch.arange(0, instance_mask.shape[1])
                    .view(1, 1, -1)
                    .expand(1, *instance_mask.shape),
                )
            ).reshape(2, -1)

            segm_pred_clustered = self.cluster.cluster_pixels(
                segm_prediction, n_sigma=2
            )
            grid = grid[:, ~torch.any(idx_seg_pred < 0, dim=0)]
            idx_seg_pred = idx_seg_pred[:, ~torch.any(idx_seg_pred < 0, dim=0)]

            grid = grid[:, ~(idx_seg_pred[0] > instance_mask.shape[0] - 1)]
            idx_seg_pred = idx_seg_pred[
                :, ~(idx_seg_pred[0] > instance_mask.shape[0] - 1)
            ]

            grid = grid[:, ~(idx_seg_pred[1] > instance_mask.shape[1] - 1)]
            idx_seg_pred = idx_seg_pred[
                :, ~(idx_seg_pred[1] > instance_mask.shape[1] - 1)
            ]

            idx_flat = np.ravel_multi_index(
                (idx_seg_pred[0].cpu(), idx_seg_pred[1].cpu()), instance_mask.shape
            )
            grid_flat = np.ravel_multi_index(
                (grid[0].cpu(), grid[1].cpu()), instance_mask.shape
            )
            centroid_pred = torch.zeros_like(
                instance_mask
            ).flatten()  # visualize where the cell pix will be clustered
            centroid_pred[idx_flat] = instance_mask.flatten()[grid_flat]
            centroid_pred = centroid_pred.reshape(instance_mask.shape)

            is_cell = instance_mask != 0
            pix_pred = torch.zeros_like(instance_mask)
            cell_pred = (segm_pred_clustered != 0).cuda()
            pix_pred[~is_cell & ~cell_pred] = 0  # bg -> bg == 0
            pix_pred[~is_cell & cell_pred] = 2  # bg-> cell == 2
            pix_pred[is_cell & ~cell_pred] = 3  # cell ->bg == 3
            pix_pred[is_cell & cell_pred] = 1  # cell ->cell == 1

            seed_map = torch.sigmoid(segm_prediction[-1])
            # mask data
            instance_mask_masked = np.ma.array(
                instance_mask.cpu(), mask=instance_mask.cpu() == 0
            )
            segm_pred_clustered_masked = np.ma.array(
                segm_pred_clustered.cpu(), mask=segm_pred_clustered.cpu() == 0
            )

            # same colors for the instances at t, t+1 -> vim, vmx for both images
            ids_prev = instance_mask_prev_frame[instance_mask_prev_frame > 0].unique()
            ids_curr = instance_mask[instance_mask > 0].unique()
            all_ids = torch.cat([ids_prev, ids_curr]).cpu().numpy()
            min_id = max_id = 0
            if len(all_ids) > 0:
                min_id = min(all_ids)
                max_id = max(all_ids)
            self.axes["segmentation"][0].imshow(
                instance_mask_masked, cmap=self.gray_cmp, vmin=min_id, vmax=max_id
            )
            self.axes["segmentation"][0].set_title(r"GT Instances $t+1$")

            self.axes["segmentation"][1].imshow(
                segm_pred_clustered_masked, cmap=self.gray_cmp
            )
            self.axes["segmentation"][1].set_title(r"Pred. Instances $t+1$")

            # show original cell shape as well -> set as -1
            centroid_pred[
                torch.logical_and(instance_mask.detach() != 0, centroid_pred == 0)
            ] = -10000
            centroid_pred_masked = np.ma.array(
                centroid_pred.cpu(), mask=centroid_pred.cpu() == 0
            )
            ids = np.unique(instance_mask.detach().cpu())
            ids = ids[ids > 0]

            self.axes["segmentation"][3].imshow(
                centroid_pred_masked, cmap=self.gray_cmp, vmin=min_id, vmax=max_id
            )
            self.axes["segmentation"][3].set_title(
                r"Offset Pred. $\mathbf{x}+\hat{\mathbf{o}}$ $t+1$"
            )
            pixel_positions = np.unravel_index(idx_flat, instance_mask.shape)
            pixel_label = instance_mask.flatten().cpu()[grid_flat].numpy()
            self.add_centroid_marker(
                self.axes["segmentation"][3],
                pixel_positions,
                pixel_label,
                (min_id, max_id),
            )

            self.axes["segmentation"][4].set_title(r"Center Prediction $t+1$")
            c_ax = self.axes["segmentation"][4].imshow(
                seed_map.cpu(),
                cmap=self.gray_cmp,
                norm=mpl.colors.Normalize(vmin=0, vmax=1),
            )
            self.add_colorbar(
                self.figures["segmentation"],
                c_ax,
                self.axes["segmentation"][4],
                ("seed_map", 4),
            )

            self.axes["segmentation"][2].imshow(
                pix_pred.cpu().squeeze(),
                cmap=self.cmap_pix_pred,
            )
            classes = [
                r"bg$\rightarrow\widehat{\mathrm{bg}}$",
                r"cell$\rightarrow\widehat{\mathrm{cell}}$",
                r"bg$\rightarrow\widehat{\mathrm{cell}}$",
                r"cell$\rightarrow\widehat{\mathrm{bg}}$",
            ]
            label = [
                mpl.patches.Patch(color=self.cmap_pix_pred(i), label=classes[i])
                for i in range(self.cmap_pix_pred.N)
            ]
            self.axes["segmentation"][2].set_title(r"Predicted Class vs GT")
            self.axes["segmentation"][2].legend(
                handles=label,
                title=r"GT$\rightarrow$Prediction",
                ncol=2,
                loc="lower left",
                fontsize=matplotlib.rcParams["font.size"],
            )
            plt.draw()
            # same scaling of other plots as for plots that have a c
            self.add_dummy_axis(self.axes["segmentation"][0])
            self.add_dummy_axis(self.axes["segmentation"][1])
            self.add_dummy_axis(self.axes["segmentation"][2])
            self.add_dummy_axis(self.axes["segmentation"][3])

            self.mypause(0.0001)

    def add_centroid_marker(self, ax, pix_position, pix_label, min_max_idx):
        """Mark gt cell centroid positions in the subplot."""
        for p in self.patches:
            p.remove()
        self.patches = []
        df = DataFrame.from_records(list(zip(pix_label, *pix_position)))
        df = df.set_index(0)  # set label as index
        mask_positions = df.groupby(0).apply(lambda x: np.median(np.array(x), 0))
        # remove -1 marker (only to visualize cell positions)
        mask_positions = mask_positions[mask_positions.index > 0]
        min_id, max_id = min_max_idx
        for m_id, m_center in mask_positions.items():
            # swap yx -> xy coordinates of the center as circle expects xy coordinates
            marker = Circle(
                m_center[::-1],
                radius=4,
                edgecolor="w",
                facecolor=self.gray_cmp((m_id - min_id) / (max_id - min_id + 1e-10)),
                linewidth=2,
            )
            ax.add_patch(marker)
            self.patches.append(marker)

    def add_dummy_axis(self, ax):
        divider = make_axes_locatable(ax)
        cb = divider.append_axes("right", size=str(self.cbar_size) + "%", pad=0.05)
        cb.remove()

    def vis_tracking(self, prediction, ground_truth, instance_mask_prev_frame):
        """Visualize the tracking prediction."""
        with torch.no_grad():
            _, offset_pred = prediction
            instance_mask, center_image, offset = ground_truth
            gt_flow = self.calc_gt_flow(instance_mask, center_image, offset)
            cell_mask = np.tile((instance_mask == 0).cpu()[np.newaxis, ...], (2, 1, 1))

            pred_flow = torch.tanh(offset_pred)

            pred_flow_masked = np.ma.array(pred_flow.cpu(), mask=cell_mask)

            # mask data
            gt_flow_masked = np.ma.array(gt_flow.cpu(), mask=cell_mask)
            instance_mask_masked = np.ma.array(
                instance_mask.cpu(), mask=instance_mask.cpu() == 0
            )
            instance_mask_prev_frame_masked = np.ma.array(
                instance_mask_prev_frame.cpu(), mask=instance_mask_prev_frame.cpu() == 0
            )

            gt_flow_mag, gt_flow_phase = self.calc_magnitude_and_phase(
                gt_flow_masked * self.grid_shape.cpu().numpy().reshape(-1, 1, 1)
            )
            mask_missing = cell_ids_without_center(instance_mask, center_image).cpu()
            gt_flow_mag[mask_missing] = -10000
            gt_flow_phase[mask_missing] = -10000
            pred_flow_mag, pred_flow_phase = self.calc_magnitude_and_phase(
                pred_flow_masked * self.grid_shape.cpu().numpy().reshape(-1, 1, 1)
            )

            # same colors for the instances at t, t+1 -> vmin, vmax for both images
            ids_prev = instance_mask_prev_frame[instance_mask_prev_frame > 0].unique()
            ids_curr = instance_mask[instance_mask > 0].unique()
            all_ids = torch.cat([ids_prev, ids_curr]).cpu().numpy()
            min_id = max_id = 0
            if len(all_ids) > 0:
                min_id = min(all_ids)
                max_id = max(all_ids)

            self.axes["tracking"][0][0].imshow(
                instance_mask_masked, cmap=self.gray_cmp, vmin=min_id, vmax=max_id
            )
            self.axes["tracking"][1][0].imshow(
                instance_mask_prev_frame_masked,
                cmap=self.gray_cmp,
                vmin=min_id,
                vmax=max_id,
            )
            self.axes["tracking"][0][0].set_title(r"GT Instances $t+1$")
            self.axes["tracking"][1][0].set_title(r"GT Instances $t$")

            c_ax = self.axes["tracking"][0][1].imshow(
                gt_flow_mag,
                cmap=self.gray_cmp,
                norm=matplotlib.colors.SymLogNorm(
                    linthresh=torch.norm(self.grid_shape.float()) / 10,
                    vmin=0,
                    vmax=torch.norm(self.grid_shape.float()),
                    base=10,
                    linscale=3,
                ),
            )
            self.add_colorbar(
                self.figures["tracking"],
                c_ax,
                self.axes["tracking"][0][1],
                ("gt_mag", 0, 1),
            )
            c_ax = self.axes["tracking"][1][1].imshow(
                gt_flow_phase,
                cmap=self.cyclic_cmap,
                norm=mpl.colors.Normalize(vmin=-1, vmax=1),
            )
            self.add_colorbar(
                self.figures["tracking"],
                c_ax,
                self.axes["tracking"][1][1],
                ("gt_phase", 1, 1),
            )
            c_ax.colorbar.set_ticks([-1, -0.5, 0, 0.5, 1], False)
            c_ax.colorbar.set_ticklabels(
                [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
            )

            self.axes["tracking"][0][1].set_title(r"GT Offset Mag.")
            self.axes["tracking"][1][1].set_title(r"GT Offset Direction")

            c_ax = self.axes["tracking"][0][2].imshow(
                pred_flow_mag,
                cmap=self.gray_cmp,
                norm=matplotlib.colors.SymLogNorm(
                    linthresh=torch.norm(self.grid_shape.float()) / 10,
                    vmin=0,
                    vmax=torch.norm(self.grid_shape.float()),
                    base=10,
                    linscale=3,
                ),
            )

            self.add_colorbar(
                self.figures["tracking"],
                c_ax,
                self.axes["tracking"][0][2],
                ("pred_mag", 0, 2),
            )
            c_ax = self.axes["tracking"][1][2].imshow(
                pred_flow_phase,
                cmap=self.cyclic_cmap,
                norm=mpl.colors.Normalize(vmin=-1, vmax=1),
            )
            self.add_colorbar(
                self.figures["tracking"],
                c_ax,
                self.axes["tracking"][1][2],
                ("pred_phase", 1, 2),
            )
            c_ax.colorbar.set_ticks([-1, -0.5, 0, 0.5, 1], False)
            c_ax.colorbar.set_ticklabels(
                [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
            )

            self.axes["tracking"][0][2].set_title(r"Pred. Offset Mag.")
            self.axes["tracking"][1][2].set_title(r"Pred. Offset Direction")

            # same scaling of other plots as for plots that have a c
            self.add_dummy_axis(self.axes["tracking"][0][0])
            self.add_dummy_axis(self.axes["tracking"][1][0])

            plt.draw()
            self.mypause(0.0001)

    def calc_magnitude_and_phase(self, flow):
        flow[1][flow[1] == 0] = 1e-10
        flow *= -1
        mag = np.sum(flow ** 2, axis=0) ** 0.5
        phase = np.arctan2(flow[0], flow[1]) / np.pi
        return mag, phase

    def calc_gt_centroid_pos(self, instance_mask, center_image, offset=None):
        """Convert position of the cell centers from pixel positions to grid positions."""
        height, width = instance_mask.shape
        yxm_s = self.grid[:, 0:height, 0:width].contiguous()  # 2 x h x w

        instance_ids = instance_mask.unique()
        instance_ids = instance_ids[instance_ids != 0]
        gt_position = torch.zeros((2, *instance_mask.shape)).double().cuda()
        for inst_id in instance_ids:
            in_mask = instance_mask.eq(inst_id)  # 1 x h x w
            index_gt_center = (in_mask & center_image.byte().bool()).squeeze()
            if index_gt_center.sum().eq(0):  # missing centroid -> set to pix position
                gt_center_position = yxm_s[:, in_mask]
            elif offset is None:  # segmentation
                gt_center_position = yxm_s[:, index_gt_center]
            else:  # tracking
                gt_center_position = (
                    yxm_s[:, index_gt_center]
                    - offset[:, index_gt_center] / self.grid_shape
                )
            gt_position[:, in_mask.squeeze()] = gt_center_position.double()
        return gt_position

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams["backend"]
        if backend in mpl.rcsetup.interactive_bk:
            figManager = mpl._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


def calc_grid(grid_x, grid_y, pixel_x, pixel_y):
    xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
    ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
    grid = torch.cat((ym, xm), 0)
    return grid


def cell_ids_without_center(instance_mask, center_mask):
    center_ids = instance_mask[center_mask].unique()
    masked_cells = ~(instance_mask[..., np.newaxis] == center_ids).any(-1) & (
        instance_mask > 0
    )
    return masked_cells


def log_to_scalar_ticks_label(tick_labels):
    """Convert logarithmic ticks label to linear ticks label."""
    for label in tick_labels:
        # label is a Text instance with (x_positon, y_position, text)
        text = label._text

        if text == "":
            continue
        if "^" not in text:
            continue
        try:
            factor, base, exponent = re.findall("\d+", text)
        except ValueError:
            base, exponent = re.findall("\d+", text)
            factor = 1
        factor = int(factor)
        base = int(base)
        exponent = int(exponent)
        number = factor * (base ** exponent)
        new_text = f"$\\mathdefault{{{number}}}$"
        label._text = new_text
    return tick_labels


def degrid(prediction, grid_size, pixel_size):
    return torch.round(prediction * (grid_size - 1) / pixel_size).int()
