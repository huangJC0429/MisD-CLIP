import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from logging import StreamHandler
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.stats as st
import torch
import yaml
from accelerate import Accelerator
from requests.adapters import HTTPAdapter
from torch import nn
from urllib3.util import Retry
from utils.util_calibration import _ECELoss, plot_acc_calibration, plot_histograms, get_AUROC_AUPR_FPR, get_metric
from models.calibration_net import CalibrationNet

from data import CustomDataset, dataset_custom_prompts

from utils import (
    Config,
    dataset_object,
    evaluate_predictions,
    get_class_names,
    get_labeled_and_unlabeled_data,
    save_parameters,
    save_predictions,
    store_results,
)

accelerator = Accelerator()

logger_ = logging.getLogger()
logger_.level = logging.INFO
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")


class AccelerateHandler(StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)

    def emit(self, record):
        if accelerator.is_local_main_process:
            super().emit(record)


stream_handler = AccelerateHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger_.addHandler(stream_handler)

log = logging.getLogger(__name__)


def workflow(dataset_dir, obj_conf):
    # Get dataset name
    # We get the dataset name from the dev_config.py
    dataset = obj_conf.DATASET_NAME
    # Get class names of target task
    # define function for each dataset
    classes, seen_classes, unseen_classes = get_class_names(dataset, dataset_dir, obj_conf.SPLIT_SEED)
    # Create dict classes to pass as variable
    dict_classes = {
        "classes": classes,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
    }
    # Log number of classes
    log.info(f"\n----------------------DATA INFO-----------------------\n")
    log.info(f"Number of classes split {obj_conf.SPLIT_SEED}: {len(classes)}")
    log.info(f"Number of seen classes split {obj_conf.SPLIT_SEED}: {len(seen_classes)}")
    log.info(f"Number of unseen classes split {obj_conf.SPLIT_SEED}: {len(unseen_classes)}")
    # Path for images
    data_folder = f"{dataset_dir}/{dataset}"
    log.info(f"Data folder: {data_folder}")
    log.info(f"\n-------------------------------------------------------------\n")
    
    # Get labeled data (seen classes)
    # Get unlabeled data (unseen classes)
    # Get test data (both seen and unseen classes)
    labeled_data, unlabeled_data, test_data = get_labeled_and_unlabeled_data(
        dataset, data_folder, seen_classes, unseen_classes, classes
    )  # 这里的unlabeled_data 就是专门留给self-training打伪标签用的,而且是unseen class
    # print(unlabeled_data)
    # exit()
    # print(len(labeled_data))
    # print(len(unlabeled_data))
    # print(len(test_data))
    # exit()

    # Create datasets
    labeled_files, labeles = zip(*labeled_data) # labeled_files:图片，labels: text
    unseen_labeled_files, unseen_labeles = zip(*unlabeled_data)
    test_labeled_files, test_labeles = zip(*test_data)
    label_to_idx = {c: idx for idx, c in enumerate(classes)}

    # print(unseen_labeles)
    # exit()

    # class to number
    label2id = {label: idx for idx, label in enumerate(classes)}
    numeric_testlabels = [label2id[label] for label in test_labeles] # 0,0,1,1,2,2,3,3,..

    # Select few-samples
    few_shots_files = []
    few_shots_labs = []

    labeled_files = np.array(labeled_files)
    # print(len(labeled_files)) 3760
    # exit()
    labeles = np.array(labeles)
    for c in seen_classes:
        np.random.seed(obj_conf.validation_seed)
        indices = np.random.choice(
            np.where(labeles == c)[0],
            size=obj_conf.N_LABEL,
            replace=False,
        )
        few_shots_files += list(labeled_files[indices])
        few_shots_labs += list(labeles[indices])

    # print(classes)
    # print(len(classes))
    # print(test_labeles)

    # print(numeric_testlabels)
    # print(test_labeles)
    # exit()

    log.info(
        f"NUMBER OF SHOTS =  {len(classes)} (NUM_CLASSES) X {obj_conf.N_LABEL} (SHOTS PER CLASS): {obj_conf.N_LABEL * len(classes)}")
    log.info(f"NUMBER OF SHOTS {len(few_shots_labs)}")

    # # Define the set of unlabeled data which excludes the few samples labeled data
    # unseen_labeled_files = []
    # unseen_labeles = []
    # for idx, f in enumerate(labeled_files):
    #     if f not in few_shots_files:
    #         unseen_labeled_files += [f]
    #         unseen_labeles += [labeles[idx]]
    #
    # log.info(f"Size of unnlabeled data: {len(unseen_labeled_files)}")

    # Define the few shots as the labeled data
    labeled_files = few_shots_files
    labeles = few_shots_labs


    # Separate train and validation
    np.random.seed(obj_conf.validation_seed)
    train_indices = np.random.choice(
        range(len(labeled_files)),
        size=int(len(labeled_files) * obj_conf.ratio_train_val),
        replace=False,
    )
    val_indices = list(set(range(len(labeled_files))).difference(set(train_indices)))

    train_labeled_files = np.array(labeled_files)[train_indices]
    train_labeles = np.array(labeles)[train_indices]

    val_labeled_files = np.array(labeled_files)[val_indices]
    val_labeles = np.array(labeles)[val_indices]

    DatasetObject = dataset_object(obj_conf.DATASET_NAME)
    # Training set (labeled seen)
    train_seen_dataset = DatasetObject( # data.dataset.DTD object
        train_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=train_labeles,
        label_map=label_to_idx,
    )
    # Training set (unlabeled unseen)
    train_unseen_dataset = DatasetObject(
        unseen_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=None,
        label_map=label_to_idx,
    )
    # Validation set (labeled seen)
    val_seen_dataset = DatasetObject(
        val_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=val_labeles,
        label_map=label_to_idx,
    )
    # Test set (test seen and unseen)
    test_dataset = DatasetObject(
        test_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=False,
        labels=None,
        label_map=label_to_idx,
    )

    # filter new classes
    filtered = [(f, l) for f, l in zip(test_labeled_files, test_labeles) if l in unseen_classes]

    new_test_labeled_files, new_test_labels = zip(*filtered) if filtered else ((), ())
    numeric_newtestlabels = [label2id[label] for label in new_test_labels]
    # print(len(new_test_labels))
    # print(len(test_labeles))
    # exit()


    test_unseen_dataset = DatasetObject(
        new_test_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=False,
        labels=None,
        label_map=label_to_idx,
    )

    # Log info data
    log.info(f"\n----------------------TRAINING DATA INFO-----------------------\n")
    log.info(f"Len training seen data: {len(train_seen_dataset.filepaths)}")
    log.info(f"Average number of labeled images per seen class:{len(train_seen_dataset.filepaths)/len(seen_classes)} ")
    log.info(f"Len training unseen data: {len(train_unseen_dataset.filepaths)}")
    log.info(f"Len validation seen data: {len(val_seen_dataset.filepaths)}")
    log.info(f"Len test data: {len(test_dataset.filepaths)}")
    log.info(f"\n-------------------------------------------------------------\n")
    # Define model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"\n----------------------MODEL INFO-----------------------\n")


    CN_model2 = CalibrationNet(obj_conf,label_to_idx, classes, obj_conf.beta, device)
    CN_model2.train_cn(train_seen_dataset, val_seen_dataset)

    df_predictions2, images2, predictions2, prob_preds2, clip_logits = CN_model2.test_cn(test_unseen_dataset)
    # zero shot clip
    auroc, aupr_cor, aupr_incor, fpr_n_cor, fpr_n_incor, a, b = get_metric(clip_logits, numeric_newtestlabels, isprint=False)
    print(f"New Class**->Zero-shot CLIP: AUROC: {auroc:.3f},AUPR_cor: {aupr_cor:.3f},"
           f"AUPR_incor: {aupr_incor:.3f},FPR95%_cor: {fpr_n_cor:.3f},FPR95%_incor: {fpr_n_incor:.3f}, FPR90%_cor: {a:.3f},FPR90%_incor: {b:.3f}")
    # Ours
    auroc, aupr_cor, aupr_incor, fpr_n_cor, fpr_n_incor, a, b = get_metric(prob_preds2, numeric_newtestlabels, isprint=False)
    print(
        f"New Class**->Ours: AUROC: {auroc:.3f},AUPR_cor: {aupr_cor:.3f},"
         f"AUPR_incor: {aupr_incor:.3f},FPR95%_cor: {fpr_n_cor:.3f},FPR95%_incor: {fpr_n_incor:.3f}, FPR90%_cor: {a:.3f},FPR90%_incor: {b:.3f}")



 
def main():
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument(
        "--model_config",
        type=str,
        default="model_config.yml",
        help="Name of model config file",
    )
    parser.add_argument(
        "--learning_paradigm",
        type=str,
        default="trzsl",
        help="Choose among trzsl, ssl, and ul",
    )

    args = parser.parse_args()

    with open(f"methods_config/our/trzsl_{args.model_config}", "r") as file:
        config = yaml.safe_load(file)

    # Cast configs to object
    obj_conf = Config(config)

    # Set seed
    optim_seed = int(os.environ["OPTIM_SEED"])
    obj_conf.OPTIM_SEED = optim_seed
    # Set backbone
    obj_conf.VIS_ENCODER = os.environ["VIS_ENCODER"]
    # Set dataset name
    obj_conf.DATASET_NAME = os.environ["DATASET_NAME"]
    # Set dataset dir
    obj_conf.DATASET_DIR = os.environ["DATASET_DIR"]
    # Set model name
    obj_conf.MODEL = os.environ["MODEL"]
    # Set split seed
    obj_conf.SPLIT_SEED = int(os.environ["SPLIT_SEED"])
    # Set dataset's template for textual prompts
    obj_conf.PROMPT_TEMPLATE = dataset_custom_prompts[obj_conf.DATASET_NAME]
    # Set data dir
    dataset_dir = obj_conf.DATASET_DIR
    # Set learning paradigm
    obj_conf.LEARNING_PARADIGM = args.learning_paradigm

    # Set tau for calibration model
    obj_conf.n_bins = 20
    obj_conf.tau = obj_conf.TAU # 0.05 # 0.05
    obj_conf.n_hid = obj_conf.N_HID # 16
    obj_conf.dropout = 0.6

    obj_conf.beta = obj_conf.BETA # for incorrect, correct sample Separation 0.8
    
    # Set the file path for the log file
    log_file = f"logs/TRZSL_{obj_conf.DATASET_NAME}_{obj_conf.MODEL}_{obj_conf.VIS_ENCODER.replace('/', '-')}.log"
    # Create a FileHandler and set the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    # Add the FileHandler to the logger
    logger_.addHandler(file_handler)

    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Dataset dir: {dataset_dir}")

    # Check dataset directory exists
    if not Path(dataset_dir).exists():
        print(dataset_dir)
        raise Exception("`dataset_dir` does not exist..")

    # Set random seeds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(obj_conf.OPTIM_SEED)
    random.seed(obj_conf.OPTIM_SEED)
    torch.manual_seed(obj_conf.OPTIM_SEED)
    accelerator.wait_for_everyone()
    # Seed for cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed(obj_conf.OPTIM_SEED)
        torch.cuda.manual_seed_all(obj_conf.OPTIM_SEED)
        accelerator.wait_for_everyone()

    torch.backends.cudnn.benchmark = True

    workflow(dataset_dir, obj_conf)


if __name__ == "__main__":
    main()
