#!/bin/bash

for dataset_dir in './data' ; do # add here the path to the folder containing dataset folders
for vis_encoder in 'ViT-B/32'; do # You can choose among 'ViT-B/32' and 'ViT-L/14'
for split_seed in 500; do # This indicate the split for TRZSL, i.e., 500, 0, or 200. For other learning setting this is 500 as default.
for dataset_name in DTD; do # DTD Flowers102 EuroSAT FGVCAircraft RESICS45 MNIST CUB
for model in textual_prompt ; do #textual_prompt visual_prompt multimodal_prompt
for optim_seed in 1; do
# for meta_lr in '0.001'; do # 0.0002
    
    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"
    export DATASET_DIR="$dataset_dir"
    # export META_LR="$meta_lr"
    
    # Set accelerate configuration file to to accelerate_config.yml when running on GPUs
    accelerate launch --config_file methods_config/accelerate_localtest_config.yml run_main_ssl_cali.py --model_config ${dataset_name}_config.yml \
                      --learning_paradigm ssl # Choose among ul, ssl, and trzsl

done
done
done
done
done
done