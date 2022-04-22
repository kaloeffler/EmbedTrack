"""Measurements to evaluate the tracking performance.
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def select_platform():
    """Selects for the cell tracking challenge measure the correct executables."""
    sys_type = platform.system()
    if sys_type == "Linux":
        return "Linux"
    if sys_type == "Windows":
        return "Win"
    if sys_type == "Darwin":
        return "Mac"
    raise ValueError("Platform not supported")


def calc_tra_score(result_dir, gt_dir, get_aogm_results=True):
    """
    Calculates TRA score using the CTC executables.
    Args:
        result_dir: posix path to the tracking directory
        gt_dir: posix path to the ground truth data directory
        get_aogm_results: bool whether to extract AOGM measures as well

    Returns: float which is the TRA score, (tracking errors (dict) if get_aogm_results is true)

    """
    tra_measure_dir = Path(__file__).parent / "CTC_eval"
    assert Path.exists(tra_measure_dir), "missing folder with CTC measures"
    platform_name = select_platform()
    tra_measure_file = [
        file
        for file in os.listdir(Path(tra_measure_dir, platform_name))
        if file.startswith("TRAMeasure")
    ]
    assert (
        len(tra_measure_file) == 1
    ), "just one tracking measure executable per sub directory allowed"
    tra_measure_path = Path(tra_measure_dir, platform_name, tra_measure_file[0])
    data_dir = gt_dir.parent
    data_set_id = gt_dir.name.split("_")[0]

    # copy results to same dir as gt path, as ctc measure executable expects results
    # and gt data to be in the same directory
    temp_dir = None
    default_res_dir = gt_dir.parent / (data_set_id + "_RES")
    if result_dir.as_posix() != default_res_dir.as_posix():

        if os.path.exists(default_res_dir):
            temp_dir = data_dir / ("temp")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            shutil.move(default_res_dir.as_posix(), temp_dir.as_posix())

        shutil.move(result_dir.as_posix(), default_res_dir.as_posix())

    # ctc executable expects time points in file names to have a fixed length
    n_files = len([f for f in os.listdir(gt_dir / "TRA") if f.endswith("tif")])
    if n_files < 1000:
        n_digits = 3
    else:
        n_digits = 4

    for f in os.listdir(default_res_dir):
        if f.startswith("mask"):
            parts = f.split(".")
            mask_id = f.split(".")[0].split("mask")[1].zfill(n_digits)
            new_name = "mask" + mask_id + "." + parts[1]
            shutil.move(default_res_dir / f, default_res_dir / new_name)

    output = subprocess.Popen(
        [tra_measure_path.as_posix(), data_dir.as_posix(), data_set_id, str(n_digits)],
        stdout=subprocess.PIPE,
    )
    result, _ = output.communicate()

    tra_measure = re.findall(r"\d\.\d*", result.decode("utf-8"))
    print(result)

    # undo moving of result folder
    print(result_dir)
    if result_dir.as_posix() != default_res_dir.as_posix():
        shutil.move(default_res_dir.as_posix(), result_dir.as_posix())
        if temp_dir is not None:
            shutil.move(temp_dir.as_posix(), default_res_dir.as_posix())

    assert len(tra_measure) == 1, "error in extraction of tra measure"
    tra_measure = float(tra_measure[0])
    tra_log_file = result_dir / "TRA_log.txt"
    assert Path.exists(tra_log_file), "TRA log file missing"
    tra_errors = extract_aogm_info(tra_log_file)

    if get_aogm_results:
        return tra_measure, tra_errors
    return tra_measure


def calc_ctc_scores(result_dir, gt_dir):
    """
    Extracts all CTC measures (DET,SEG,TRA) using the CTC executables.
    Args:
        result_dir: posix path to the tracking directory
        gt_dir: posix path to the ground truth data directory

    Returns: directory containing the metric names and their scores

    """

    ctc_measure_dir = Path(__file__).parent / "CTC_eval"
    assert Path.exists(ctc_measure_dir), "missing folder with CTC measures"
    platform_name = select_platform()
    measure_files = [
        file
        for file in os.listdir(Path(ctc_measure_dir, platform_name))
        if "Measure" in file
    ]

    data_dir = gt_dir.parent
    data_set_id = gt_dir.name.split("_")[0]
    # copy results to same dir as gt path, as ctc measure executable expects results
    # and gt data to be in the same directory
    temp_dir = None
    default_res_dir = data_dir / (data_set_id + "_RES")
    if result_dir.as_posix() != default_res_dir.as_posix():

        if os.path.exists(default_res_dir):
            temp_dir = data_dir / ("temp")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            shutil.move(default_res_dir.as_posix(), temp_dir.as_posix())

        shutil.move(result_dir.as_posix(), default_res_dir.as_posix())
    # ctc executable expects time points in file names to have a fixed length
    n_files = max(
        [
            int(re.findall(r"\d+", f)[0])
            for f in os.listdir(gt_dir / "TRA")
            if f.endswith("tif")
        ]
    )
    if n_files < 1000:
        n_digits = 3
    else:
        n_digits = 4

    for f in os.listdir(default_res_dir):
        if f.startswith("mask"):
            parts = f.split(".")
            mask_id = f.split(".")[0].split("mask")[1].zfill(n_digits)
            new_name = "mask" + mask_id + "." + parts[1]
            shutil.move(default_res_dir / f, default_res_dir / new_name)

    all_results = {}
    for measure_f in measure_files:
        if not os.path.exists(
            default_res_dir / "res_track.txt"
        ) and measure_f.startswith("TRA"):
            continue
        ctc_metric_path = Path(ctc_measure_dir, platform_name, measure_f)
        output = subprocess.Popen(
            [
                ctc_metric_path.as_posix(),
                data_dir.as_posix(),
                data_set_id,
                str(n_digits),
            ],
            stdout=subprocess.PIPE,
        )
        result, _ = output.communicate()

        metric_score = re.findall(r"\d\.\d*", result.decode("utf-8"))
        all_results[measure_f[:3]] = metric_score

        print(result)

    # undo moving of result folder
    print(result_dir)
    if result_dir.as_posix() != default_res_dir.as_posix():
        shutil.move(default_res_dir.as_posix(), result_dir.as_posix())
        if temp_dir is not None:
            shutil.move(temp_dir.as_posix(), default_res_dir.as_posix())

    for metric_name, score in all_results.items():
        assert len(score) == 1, f"error in extraction of {metric_name} measure"
    all_results = {
        metric_name: float(score[0]) for metric_name, score in all_results.items()
    }
    return all_results


def extract_tra_score_from_log(file):
    return extract_metric_score_from_log(file, "TRA")


def extract_metric_score_from_log(file, metric_name):
    """Extracts a ctc metric score from log file."""
    pattern = re.compile(f"{metric_name} measure: (\d+\.\d*).*")

    with open(file, "r") as f:
        for line in f.readlines():
            if pattern.match(line):
                return float(pattern.match(line).group(1))


def extract_aogm_info(tra_log_file):
    """Extracts from the TRA log file the number of different tracking errors.

    Args:
        tra_log_file: path to the tra log file

    Returns: dict containing tracking errors overall count and over time

    """
    pattern = re.compile("\[*T=\d+")
    with open(tra_log_file, "r") as f:
        lines = f.readlines()

        error_counts = {
            "EdgeSplit": 0,
            "EdgeAdd": 0,
            "EdgeDelete": 0,
            "EdgeSemantic": 0,
            "FP": 0,
            "FN": 0,
        }

        errors_over_t = {
            "EdgeSplit": {},  # time id: value
            "EdgeAdd": {},
            "EdgeDelete": {},
            "EdgeSemantic": {},
            "FP": {},
            "FN": {},
        }
        counter_key = None
        for line in lines:
            match = re.match(pattern, line)
            if match is not None and counter_key is not None:
                time_id = int(match.group().split("=")[-1])
                if time_id not in errors_over_t[counter_key].keys():
                    errors_over_t[counter_key][time_id] = 0
                errors_over_t[counter_key][time_id] += 1

            if "Splitting" in line:
                counter_key = "EdgeSplit"
            elif "False Negative Vertices" in line:
                counter_key = "FN"
            elif "False Positive Vertices" in line:
                counter_key = "FP"
            elif "Redundant Edges" in line:
                counter_key = "EdgeDelete"
            elif "Edges To Be Added" in line:
                counter_key = "EdgeAdd"
            elif "Edges with Wrong Semantics" in line:
                counter_key = "EdgeSemantic"
            elif line.startswith("==="):
                break
            else:
                if counter_key is not None:
                    error_counts[counter_key] += 1
        aogm_measure = {"counts": error_counts, "t": errors_over_t}

    return aogm_measure
