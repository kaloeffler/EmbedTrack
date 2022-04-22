"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""

import os
from datetime import datetime
from pathlib import Path
from time import time

from embedtrack.infer.infer_ctc_data import inference

FILE_PATH = Path(__file__)
PROJECT_PATH = os.path.join(*FILE_PATH.parts[:-3])

RAW_DATA_PATHS = [os.path.join(PROJECT_PATH, "ctc_raw_data/challenge"),
                  os.path.join(PROJECT_PATH, "ctc_raw_data/train")]
MODEL_PATH = os.path.join(PROJECT_PATH, "models")
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

CALC_CTC_METRICS = False

# Adam optimizer; normalize images; OneCycle LR sheduler; N epochs
MODEL_NAME = "adam_norm_onecycle_15"
BATCH_SIZE = 32
runtimes = {}
for raw_data_path in RAW_DATA_PATHS:
    for data_set in DATA_SETS:
        for data_id in ["01", "02"]:
            img_path = os.path.join(raw_data_path, data_set, data_id)

            model_dir = os.path.join(MODEL_PATH, data_set, MODEL_NAME)
            if not os.path.exists(model_dir):
                print(f"no trained model for data set {data_set}")
                continue

            # time stamps
            timestamps_trained_models = [
                datetime.strptime(time_stamp, "%Y-%m-%d---%H-%M-%S")
                for time_stamp in os.listdir(model_dir)
            ]
            timestamps_trained_models.sort()
            last_model = timestamps_trained_models[-1].strftime("%Y-%m-%d---%H-%M-%S")
            model_path = os.path.join(model_dir, last_model, "best_iou_model.pth")
            config_file = os.path.join(model_dir, last_model, "config.json")
            t_start = time()
            inference(img_path, model_path, config_file, batch_size=BATCH_SIZE)
            t_end = time()

            run_time = t_end - t_start
            print(f"Inference Time {img_path}: {run_time}s")

