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

from data import CustomDataset, dataset_custom_prompts
from methods.semi_supervised_learning import (
    TextualPrompt,
    VisualPrompt,
)
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
from models.FTbased_calibration_net import CalibrationNet
from models.visual_FTbased_calibration_net import CalibrationNet as Visual_CalibrationNet


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
    # print(classes)
    # print(seen_classes)
    # print(unseen_classes)
    # exit()
    # We set seen and unseen to classes, since are not in th trzsl setting
    seen_classes = classes
    unseen_classes = classes
    # Create dict classes
    dict_classes = {
        "classes": classes,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
    }
    # Log number of classes
    log.info(f"\n----------------------DATA INFO-----------------------\n")
    log.info(f"Number of classes {obj_conf.SPLIT_SEED}: {len(classes)}")
    # Path for images
    data_folder = f"{dataset_dir}/{dataset}"
    log.info(f"Data folder: {data_folder}")
    log.info(f"\n-------------------------------------------------------------\n")
    
    # Get data 
    labeled_data, unlabeled_data, test_data = get_labeled_and_unlabeled_data(
        dataset, data_folder, seen_classes, unseen_classes, classes
    ) # list, len(unlabel_data) = 0, 因为是SSL设置

    # print(len(labeled_data))
    # print(len(unlabeled_data))
    # print(len(test_data))
    # exit()

    # From labeled data of all the target classes we sample few-examples
    labeled_files, labeles = zip(*labeled_data)
    test_labeled_files, test_labeles = zip(*test_data)
    label_to_idx = {c: idx for idx, c in enumerate(classes)}
    # Select few-samples
    train_few_shots_files = []
    train_few_shots_labs = []

    labeled_files = np.array(labeled_files)
    labeles = np.array(labeles)
    for c in classes:
        np.random.seed(obj_conf.validation_seed)
        indices = np.random.choice(
            np.where(labeles == c)[0], 
            size=obj_conf.N_LABEL, # the first N_LABEL for fine-tuning, te last for calibration
            replace=False, 
        )
        train_few_shots_files += list(labeled_files[indices])
        train_few_shots_labs += list(labeles[indices])

    log.info(f"NUMBER OF SHOTS =  {len(classes)} (NUM_CLASSES) X {obj_conf.N_LABEL} (SHOTS PER CLASS): {obj_conf.N_LABEL*len(classes)}")
    log.info(f"NUMBER OF SHOTS {len(train_few_shots_labs)}")
    
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
    labeled_files = train_few_shots_files
    labeles = train_few_shots_labs

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



    label2id = {label: idx for idx, label in enumerate(classes)}
    # 转换
    numeric_testlabels = [label2id[label] for label in test_labeles]

    DatasetObject = dataset_object(obj_conf.DATASET_NAME)
    # Labeled training set
    train_seen_dataset = DatasetObject(
        train_labeled_files,
        data_folder,
        transform=None, # Set later 
        augmentations=None,
        train=True,
        labels=train_labeles,
        label_map=label_to_idx,
    )

    # Validation set (labeled data)
    val_seen_dataset = DatasetObject(
        val_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=val_labeles,
        label_map=label_to_idx,
    )

    # Test set 
    test_dataset = DatasetObject(
        test_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=False,
        labels=None,
        label_map=label_to_idx,
    )
    # Log info data
    log.info(f"\n----------------------TRAINING DATA INFO-----------------------\n")
    log.info(f"Size labeled data: {len(train_seen_dataset.filepaths)}")
    log.info(f"Size validation data: {len(val_seen_dataset.filepaths)}")
    log.info(f"Size test data: {len(test_dataset.filepaths)}")
    log.info(f"\n-------------------------------------------------------------\n")
    # Define model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"\n----------------------MODEL INFO-----------------------\n")
    if obj_conf.MODEL == "visual_prompt":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = VisualPrompt(
            obj_conf, 
            label_to_idx, 
            device=device, 
            **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, 
            val_seen_dataset, 
            only_seen=True
        )

    elif obj_conf.MODEL == "textual_prompt":
        log.info(f"The model in use is: {obj_conf.MODEL}")
        model = TextualPrompt(
            obj_conf, 
            label_to_idx, 
            device=device, 
            **dict_classes
        )
        val_accuracy, optimal_prompt = model.train(
            train_seen_dataset, 
            val_seen_dataset, 
            only_seen=True
        )

    if obj_conf.MODEL != 'clip_baseline':
        # Save prompt
        save_parameters(optimal_prompt, obj_conf)
        
    # Validate on test set (standard)
    std_predictions, test_logits = model.test_predictions(test_dataset, standard_zsl=True)

    std_response = evaluate_predictions(
        obj_conf,
        std_predictions,
        test_labeled_files,
        test_labeles,
        unseen_classes,
    )


    log.info(f"ZSL accuracy: {std_response}")

    log.info(f"This is original metric score of VLM:")
    get_metric(test_logits, numeric_testlabels)

    # Training Meta net
    if obj_conf.MODEL == 'textual_prompt':
        CN_model = CalibrationNet(obj_conf, label_to_idx, classes, obj_conf.beta, model, device)
    elif obj_conf.MODEL == 'visual_prompt':
        CN_model = Visual_CalibrationNet(obj_conf, label_to_idx, classes, obj_conf.beta, model, device)
    CN_model.train_cn(val_seen_dataset, train_seen_dataset)
    df_predictions, images, predictions, prob_preds, clip_logits = CN_model.test_cn(test_dataset)

    prob_preds = prob_preds

    ece_criterion = _ECELoss(obj_conf.n_bins).cuda()
    ece = ece_criterion(prob_preds, numeric_testlabels).item()
    # print(ece)
    # plot_acc_calibration(prob_preds, numeric_testlabels, obj_conf.n_bins,
    #                      'CLIP-%s' % (obj_conf.DATASET_NAME))

    confidence_t = torch.softmax(prob_preds, dim=1).cpu().detach()
    confidence_t = torch.max(confidence_t, 1)[0]
    pred_label = torch.max(prob_preds, 1)[1]
    # correct_index_t = numeric_testlabels == pred_label.tolist()
    correct_index_t = [a == b for a, b in zip(pred_label.tolist(), numeric_testlabels)]

    # measure discrimination ability
    get_AUROC_AUPR_FPR(confidence_t, correct_index_t)

 
def main():
    parser = argparse.ArgumentParser(description="Run task")
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

    if os.environ["MODEL"] == 'textual_prompt':
        with open(f"methods_config/our/ft_ssl_{args.model_config}", "r") as file:
            config = yaml.safe_load(file)
    elif os.environ["MODEL"] == 'visual_prompt':
        with open(f"methods_config/our/visual_ft_ssl_{args.model_config}", "r") as file:
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
    # if obj_conf.DATASET_NAME == 'Flowers102':
    #     obj_conf.N_LABEL = 2
    # Set dataset dir
    obj_conf.DATASET_DIR = os.environ["DATASET_DIR"]
    # Set model name
    obj_conf.MODEL = os.environ["MODEL"]
    # Set split seed
    obj_conf.SPLIT_SEED = int(os.environ["SPLIT_SEED"])
    # Define dataset's template for textual prompts
    obj_conf.PROMPT_TEMPLATE = dataset_custom_prompts[obj_conf.DATASET_NAME]
    # Set data dir
    dataset_dir = obj_conf.DATASET_DIR
    # Set learning paradigm
    obj_conf.LEARNING_PARADIGM = args.learning_paradigm
    # # Set calibration model name
    # obj_conf.source_model = os.environ["CALIB_MODEL"]

    # Set bin for ECE loss
    obj_conf.n_bins = 20
    # Set tau for calibration model
    obj_conf.tau = obj_conf.TAU # 0.05 # 0.05
    obj_conf.n_hid = obj_conf.N_HID # 16
    obj_conf.dropout = 0.6

    obj_conf.beta = obj_conf.BETA # for incorrect, correct sample Separation 0.8
    
    # Set the file path for the log file
    log_file = f"logs/{obj_conf.DATASET_NAME}_{obj_conf.MODEL}_{obj_conf.VIS_ENCODER.replace('/', '-')}.log"
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
