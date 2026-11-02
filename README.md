# Railway-Data-Augmentation
Welcome to the accompagning GitHub repository to our paper "Does Synthetic Augmentation Help Adversarial Training?
A Railway Obstacle Classification Case Study".

## Install Dependencies

Getting started: To create an Anaconda environment with the necessary dependencies follow these steps:

```
conda env create -f environment.yml
```

## Download Dataset
Download the UAV-RSOD dataset for training on the original images here: https://zenodo.org/records/12606374
Partition the dataset using split_orig_dataset.py using classes.xlsx


## Classifier Training

Launch training of the ConvNeXt Large by executing the following command:

```bash
python src/adversarial_training.py
```

## Project Organization


    ├── README.md                   <- The top-level README for developers using this project.
    │
    ├── config                      <- Hydra condiguration files for model fine-tuning and label-to-index mapping
    │   ├── model                   <- Model configuration
    │   │   └── cnx.yaml            <- ConvNeXt Large
    │   │   └── cnxs.yaml           <- ConvNeXt Small
    │   └── label2idx.json          <- Label-to-index mapping of dataset classes
    │
    ├── Dataset-examples            <- Folder containing example images of the Synthetic-llm dataset.
    │   └── Branch                  <- Images of the class Branch
    │   └── Barrel                  <- Images of the class Barrel
    │   └── etc..                   <- Other classes
    │
    ├── dataset_original.zip        <- The original images of the UAV-RSOD dataset, partitioned and in PyTorch Dataset format. See License details below.
    │
    ├── environment.yml             <- The requirements file for reproducing the analysis environment.
    │
    ├── classes.xlsx               <- Classes of the UAV-RSOD dataset
    │
    └──  src                           <- Source code for use in this project.
         └── adversarial_training.py   <- Script for adversarial training.
         └── caption_baseline.py       <- Script for generating the rule-based prompts
         └── caption_variation.py      <- Script for generating and varying prompts using LLMs
         └── create_corruptions.py     <- Script for creating corruptions for the extended testing set
         └── evaluate_aa.py            <- Script for evaluating adversarial accuracy
         └── flux_submit.py            <- Generates images with Flux 2 by sending them on an HPC in batches using submitit
         └── trades.py                 <- trades method for adversarial training
         └── train_flux.py             <- Fine-tuning Flux
         └── utils.py                  <- Utility module for model fine-tuning
         └── models
            └── model_CNX.py           <- ConvNeXt Large architecture
            └── model_CNXs.py          <- ConvNeXt Small architecture

