"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""
import numpy as np
import tifffile

from embedtrack.utils.utils import get_img_files


def calc_cell_movement(segm_path):
    all_overlaps = []
    segm_masks = get_img_files(segm_path)
    time_points = sorted(list(segm_masks.keys()))
    # calc IOU! if same mask in both time steps! ST is somehow tracked but without lineage!
    for i_time, t_curr in enumerate(time_points[1:]):
        t_prev = time_points[i_time]
        overlaps = calc_overlap(
            tifffile.imread(segm_masks[t_curr]), tifffile.imread(segm_masks[t_prev])
        )
        all_overlaps.extend(overlaps)
    return all_overlaps


def calc_percentiles(overlaps, percentiles):
    return [np.percentile(overlaps, percentile) for percentile in percentiles]


def calc_overlap(mask_curr, mask_prev):
    cell_overlap = []
    mask_idx_curr = np.unique(mask_curr)
    mask_idx_curr = mask_idx_curr[mask_idx_curr != 0]
    for m_id in mask_idx_curr:
        if m_id in mask_prev:
            intersection = np.sum((mask_curr == m_id) & (mask_prev == m_id))
            min_cell_size = min(np.sum((mask_curr == m_id)), np.sum(mask_prev == m_id))
            cell_overlap.append(intersection / min_cell_size)
    return cell_overlap


if __name__ == "__main__":
    import os
    from pathlib import Path
    import pandas as pd

    # data configs
    FILE_PATH = Path(__file__)
    PROJECT_PATH = os.path.join(*FILE_PATH.parts[:-3])
    RAW_DATA_PATH = os.path.join(PROJECT_PATH, "train")
    RES_PATH = os.path.join(PROJECT_PATH, "results")

    DATA_SETS = [
        "Fluo-N2DH-SIM+",
        "Fluo-C2DL-MSC",
        "Fluo-N2DH-GOWT1",
        "PhC-C2DL-PSC",
        "BF-C2DL-HSC",
        "Fluo-N2DL-HeLa",
        "BF-C2DL-MuSC",
        "DIC-C2DH-HeLa",
        "PhC-C2DH-U373",
    ]
    BATCH_SIZE = 32
    all_cell_data = []
    percentiles = [5, 10, 25, 50, 75]
    for data_set in DATA_SETS:
        print(data_set)
        for data_id in ["01", "02"]:
            img_path = os.path.join(RAW_DATA_PATH, data_set, data_id)
            if data_set == "Fluo-N2DH-SIM+":
                seg_gt = "GT"
            else:
                seg_gt = "ST"
            segm_path = os.path.join(
                RAW_DATA_PATH, data_set, data_id + "_" + seg_gt, "SEG"
            )
            overlaps = calc_cell_movement(Path(segm_path))
            overlap_percentiles = calc_percentiles(overlaps, percentiles)
            cell_data = {str(k): v for k, v in zip(percentiles, overlap_percentiles)}
            cell_data.update({"Data Set": data_set, "Data Id": data_id})

            # add n frames; n_tracks; min cells, max cells
            lineage_file = os.path.join(
                RAW_DATA_PATH, data_set, data_id + "_GT", "TRA", "man_track.txt"
            )
            lineage = pd.read_csv(lineage_file, sep=" ", header=None)
            tra_files = get_img_files(Path(os.path.dirname(lineage_file)))
            first_frame = tifffile.imread(tra_files[min(tra_files.keys())])
            last_frame = tifffile.imread(tra_files[max(tra_files.keys())])
            n_idx_first = np.unique(first_frame)
            n_idx_first = n_idx_first[n_idx_first > 0]
            n_idx_last = np.unique(last_frame)
            n_idx_last = n_idx_last[n_idx_last > 0]
            cell_data.update(
                {
                    "N Tracks": len(lineage),
                    "N Frames": len(tra_files),
                    "N Cells t_start": len(n_idx_first),
                    "N Cells t_end": len(n_idx_last),
                }
            )

            print(cell_data)
            all_cell_data.append(cell_data)
    df = pd.DataFrame(all_cell_data)
    # the mean here is useless but results in the nice shaping
    sorted_columns = [
        "Data Set",
        "Data Id",
        "N Frames",
        "N Tracks",
        "N Cells t_start",
        "N Cells t_end",
        *[str(p) for p in percentiles],
    ]
    df = df.reindex(sorted_columns, axis=1)
    df = df.groupby(["Data Set", "Data Id"]).mean()
    df = df.round(3)
    with open(os.path.join(RES_PATH, "cell_overlap.tex"), "w") as file:
        print(df.to_latex(), file=file)
    with open(os.path.join(RES_PATH, "cell_overlap.txt"), "w") as file:
        print(df.to_string(), file=file)
