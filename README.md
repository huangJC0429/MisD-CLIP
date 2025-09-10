# Revisiting Calibration for Misclassification Detection in Vision-Language Models



## Table of Contents

- [Setup](#setup)
- [Data](#data)
- [Experiments](#experiments)

## Setup

To set up the project environment, follow these steps:

    ```bash
    cd MDA-CLIP/
    bash setup.sh
    ```

[Back to Table of Contents](#table-of-contents)

## Data

The experiments are conducted on the following six datasets: **D**TD **F**lowers102,  **E**uroSAT, **R**ECSIS45, **M**NIST,, and **C**UB. We use the train and test splits provided in the paper [ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models](https://openreview.net/pdf?id=hGl8rsmNXzs).

To access the FRAMED dataset, you can download it [here](https://drive.google.com/file/d/1_ns7regg8dfAAGmYcmCXa5ryuJeOoug-/view?usp=share_link). After downloading, unzip the folder to obtain the required data.

If you encounter any issues with the download or prefer an alternative method, you can follow these steps:

1. Download the data by following the instructions provided [here](https://github.com/Computer-Vision-in-the-Wild/DataDownload).
2. Rename the folders as follows:
   - `dtd/` to `DTD/`
   - `eurosat_clip/` to `EuroSAT/`
   - `oxford-flower-102/` to `Flowers102/`
   - `mnist/` to `MNIST/`
   - `resisc45_clip/` to `RESICS45/`
3. Ensure that each folder contains the following files:
   - `DTD/` should contain the [`class_names.txt`](https://github.com/BatsResearch/menghini-neurips23-code/blob/main/data/class_files/DTD/class_names.txt) file
   - `EuroSAT/` should contain the [`class_names.txt`](https://github.com/BatsResearch/menghini-neurips23-code/blob/main/data/class_files/EuroSAT/class_names.txt) file
   - `Flowers102/` should contain the [`class_names.txt`](https://github.com/BatsResearch/menghini-neurips23-code/blob/main/data/class_files/Flowers102/class_names.txt) file
   - `MNIST/` should contain the [`labels.txt`](https://github.com/BatsResearch/menghini-neurips23-code/blob/main/data/class_files/MNIST/labels.txt) file

[Back to Table of Contents](#table-of-contents)

## Experiments

Before running the experiments, create the following folders to save prompts, and results.

```bash
mkdir logs
mkdir trained_prompts
mkdir evaluation
```


### Baselines

1. CLIP 
    ```
    bash scripts/run_clip.sh
    ```
2. Pretrained CLIP based misclassification detection.
    - For few-shot setting:
        ```
        bash scripts/our_scripts/run_ssl_cali.sh
        ```
    - For new class:
        ```
        bash scripts/our_scripts/run_trzsl_cali.sh
        ```
3. Prompt-tuning CLIP based misclassification detection.
      ```
        bash scripts/our_scripts/run_FTbased_ssl_cali.sh
        ```
