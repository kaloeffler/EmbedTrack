[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<h1 align="center">EmbedTrack</h1>
<h2 align="center">Simultaneous Cell Segmentation and Tracking Through Learning Offsets and Clustering Bandwidths</h2>

<p align="center">
  <img alt="Fluo-N2DH-SIM+" src="./gifs/Fluo-N2DH-SIM+-02.gif" width="300">
  <br>
    <em>Fluo-N2DH-SIM+ tracked with EmbedTrack</em>
</p>


## Table of Contents

- **[Introduction](#introduction)**
- **[Dependencies](#dependencies)**
- **[Getting Started](#getting-started)**
- **[Datasets](#datasets)**
- **[Training and Inference on CTC Datasets](#training-and-inference-on-ctc-datasets)**
- **[CTC Trained Models and Executables](#ctc-trained-models-and-executables)**
- **[CTC Metrics](#ctc-metrics)**
- **[Citation](#citation)**
- **[Acknowledgements](#acknowledgements)**


### Introduction
This repository provides the version of the code used for the **[preprint](TODO)** **EmbedTrack - Simultaneous Cell Segmentation and Tracking Through Learning Offsets and Clustering Bandwidths**. 
Based on the instance segmentation approach of **[Neven et al., 2019](https://arxiv.org/abs/1906.11109)** and its application to cell segmentation by **[Lalit et al., 2021](https://arxiv.org/abs/2101.10033)**, we extend the instance segmentation idea to learning the task of cell segmentation and tracking jointly.
We refer to our approach as `EmbedTrack`, with which we obtain as team KIT-Loe-GE several state-of-the-art results on 2D datasets of the **[Cell Tracking Challenge](http://celltrackingchallenge.net)**.


### Dependencies 
We have tested this implementation using `pytorch` version 1.9.1 and `cudatoolkit` version 10.2 on a `linux` OS machine. 

To create the environment `venv_embedtrack` please run
```
conda env create -f path/to/environment.yml
```

### Datasets
We trained and evaluated our approach on 2D data sets from the **[Cell Tracking Challenge (CTC)](http://celltrackingchallenge.net)**:
"Fluo-N2DH-SIM+",
"Fluo-C2DL-MSC",
"Fluo-N2DH-GOWT1",
"PhC-C2DL-PSC",
"BF-C2DL-HSC",
"Fluo-N2DL-HeLa",
"BF-C2DL-MuSC",
"DIC-C2DH-HeLa", and
"PhC-C2DH-U373"-- all can be retrieved from [here](http://celltrackingchallenge.net/2d-datasets/).

### Training and Inference on CTC Datasets
To train on the CTC datasets please download and unzip the aforementioned datasets from the CTC from [here](http://celltrackingchallenge.net/2d-datasets/). Save the training data (without any renaming) of each cell type, called **Training Dataset** on the CTC website, to the folder: 
```
...\ctc_raw_data\train
```
and save the challenge datasets of each cell type, called **Challenge Dataset** on the CTC website, to the folder:
```
...\ctc_raw_data\challenge
```
After that you should have a structure in the `ctc_raw_data` folder as follows:

```
ctc_raw_data
└───train
    |───BF-C2DL-HSC
        └───01
            └───0000.tif
            ....
        └───01_GT
        └───01_ST
        ───02
            └───0000.tif
            ....
        └───02_GT
        └───02_ST
        
    └───BF-C2DL-MuSC
    ......
    
└───challenge
    |───BF-C2DL-HSC
        └───01
            └───0000.tif
            ....
        ───02
            └───0000.tif
            ....
            
    └───BF-C2DL-MuSC
    ......
    
```
Then run the script `train_all.py` to train models for joint cell segmentation and tracking for each
dataset using similar settings as in our preprint. 
The training script already includes preparing the raw CTC data for training EmbedTrack such as cropping, splitting the training sequences into a 90/10 train/val split. The data modified for training EmbedTrack is
saved in the `data` folder, and the trained models are saved in the `models` folder.

After running the training script to train models, run `infer_all.py` to generate predictions in the CTC format. The predictions are saved to the `results`folder. 

###

### CTC Trained Models and Executables
We participated as team KIT-Loe-GE on the CTC and provide executables and trained models for each of the datasets we participated on [here](http://celltrackingchallenge.net/participants/KIT-Loe-GE/).
The executables where tested on Ubuntu 16.04 and 18.04.

#### CTC Metrics
You can download the CTC metrics from **[here](http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip)** unzip it and save everything in `embedtrack/ctc_metrics/CTC_eval`. To evaluate the predictions using CTC metrics please refer to the `calc_ctc_score` function in `embedtrack/ctc_metrics/eval_ctc`.

### Citation
If you use our work in your research, please cite:

```bibtex
@misc{loeffler2022embedtrack,
      title={EmbedTrack -- Simultaneous Cell Segmentation and Tracking Through Learning Offsets and Clustering Bandwidths}, 
      author={Katharina Löffler and Ralf Mikut},
      year={2022},
      eprint={TODO},
      archivePrefix={arXiv},
      primaryClass={TODO}
}
```

### Acknowledments
Our code is based on the repositories of **[Neven et al.](https://github.com/davyneven/SpatialEmbeddings)** and **[Lalit et al.](https://github.com/juglab/EmbedSeg)** which we would like to thank for making their code publicly available.
