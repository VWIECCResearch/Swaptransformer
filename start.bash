#!/bin/bash

# Data Genration
python3 prog_caller.py  --proc EXPERT \
                        --record-data \
                        --initials PA \
                        --milestone LaneFollowing \
                        --num-eps 2 \
                        --maxstep 1000 \
                        --exec-path /opt/ml/SimPilotSimServer.x86_64 \
                        --rawdata-path /opt/ml/data

# Preprocessing the data
python3 prog_caller.py --proc PREPROCESS \
                       --initials PA \
                       --milestone LaneFollowing \
                       --rawdata-path /opt/ml/data \
                       --processeddata-path /opt/ml/processed

# Training
python3 prog_caller.py  --proc TRAIN \
                        --num-epoch 5 \
                        --training-df-path /opt/ml/processed/.pkl \
                        --training-image-path /opt/ml/processed/images \
                        --model-name pedram_model.pth \
                        --wandb-name GAN \
                        --track \
                        --saved-model \

python3 prog_caller.py --proc INFERENCE --visu


# *****************************************************
# M15 - Overtake
# *****************************************************
# Data Genration - RuleBased Driver
python prog_caller.py   --proc EXPERT \
                        --record-data \
                        --initials PA \
                        --milestone M23 \
                        --task Overtake_FMD \
                        --num-eps 250 \
                        --maxstep 7000 \
                        --exec-path /home/shamsal/Downloads/ExecutablePackage/SimPilotExe_V10_3_0_B140_Highway3LanesOneway/SimPilotSimServer \
                        --controller TravelAssistUnsafe \
                        --sumo \
                        --LaneIDSensor \
                        --rule-based \
                        --randomization-env \
                        --rand-num-vehicles 0 20 \
                        --new-rand-eps 4 \
                        --rawdata-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/raw/ \
                        --img-height 100 \
                        --img-width 50 \
                        --bev-size 50 \
                        --bevoffsetx 10 \
                        --bevoffsety 0 \
                        --track \
                        --spawnpoints-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/scenedata/newspawnpoints.csv

# Data Genration - Sumo Driver
python prog_caller.py   --proc EXPERT \
                        --record-data \
                        --initials PA \
                        --milestone M23 \
                        --task Overtake_FMD \
                        --num-eps 250 \
                        --maxstep 7000 \
                        --exec-path /home/shamsal/Downloads/ExecutablePackage/SimPilotExe_V10_3_0_B140_Highway3LanesOneway/SimPilotSimServer \
                        --controller TravelAssist \
                        --sumo \
                        --LaneIDSensor \
                        --rule-based \
                        --randomization-env \
                        --rand-num-vehicles 0 20 \
                        --new-rand-eps 4 \
                        --rawdata-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/raw/ \
                        --img-height 100 \
                        --img-width 50 \
                        --bev-size 50 \
                        --bevoffsetx 10 \
                        --bevoffsety 0 \
                        --track \
                        --spawnpoints-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/scenedata/newspawnpoints.csv


# Preprocessing the data
python prog_caller.py   --proc PREPROCESS \
                        --initials PA \
                        --milestone M23 \
                        --task Overtake_FMD \
                        --rawdata-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/raw_400_200/ \
                        --processeddata-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/processed_400_200 \
                        --sim-steptime 20 \
                        --pose-steptime 500 \
                        --num-poses 5 \
                        --compress \
                        --compresseddata-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/Compressed \
                        --compress-name compressed_Ali_400_200_2023_04_03.tar.gz

# TRAINING the data
python prog_caller.py   --proc LIGHTNING \
                        --initials PA \
                        --milestone M23 \
                        --task Overtake_FMD \
                        --bezier \
                        --travelassist-pred \
                        --img-height 100 \
                        --img-width 50 \
                        --algo BC \
                        --training-df-path /opt/data/Overtake_merged3/processed/df_balanced.pkl \
                        --training-image-path /opt/data/Overtake_merged3/processed/images \
                        --dim-input-feature 6 \
                        --num-epoch 200 \
                        --batch-size 256 \
                        --train-split 0.98 \
                        --val-starting-point 0.5 \
                        --encoder resnet18 \
                        --lr-bc 0.0005 \
                        --activation LeakyReLU \
                        --residual \
                        --track \
                        --wandb-entity icc-aitp \
                        --save-model \
                        --model-path /opt/ml/saved_models_BC \
                        --model-saverate 1 \
                        --num-workers 8

# LIGHTNING the data
python prog_caller.py   --proc LIGHTNING \
                        --initials PA \
                        --milestone M23 \
                        --task Overtake_FMD \
                        --bezier \
                        --travelassist-pred \
                        --img-height 100 \
                        --img-width 50 \
                        --algo BC \
                        --print-flag \
                        --training-df-path /opt/data/M19-Overtake_performance_large/df.pkl \
                        --training-image-path /opt/data/M19-Overtake_performance_large/images \
                        --dim-input-feature 6 \
                        --num-epoch 500 \
                        --batch-size 256 \
                        --encoder custom \
                        --lr-bc 0.0005 \
                        --activation ReLU \
                        --residual \
                        --mhsa \
                        --track \
                        --wandb-entity icc-aitp \
                        --save-model \
                        --model-path /opt/ml/saved_models_BC_M19_2023_05_25_mhsa \
                        --num-workers 4 \
                        --num-gpus 6

# LIGHTNING the data - Local
python prog_caller.py   --proc LIGHTNING \
                        --initials AS \
                        --milestone M23 \
                        --task Overtake_FMD \
                        --bezier \
                        --travelassist-pred \
                        --img-height 100 \
                        --img-width 50 \
                        --algo BC \
                        --print-flag \
                        --training-df-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/processed_M19/processed_AS_M19_Overtake_Performance_2023_05_09-09_56.pkl \
                        --training-image-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/processed_M19/images \
                        --dim-input-feature 6 \
                        --num-epoch 600 \
                        --batch-size 512 \
                        --encoder custom \
                        --lr-bc 0.0005 \
                        --activation ReLU \
                        --residual \
                        --mhsa \
                        --track \
                        --wandb-entity icc-aitp \
                        --save-model \
                        --model-path /home/shamsal/Downloads/saved_models_BC_M19_2023_05_24 \
                        --num-workers 16 \
                        --num-gpus 1


# TESTING THE TRAINING the data (SMALL)
python prog_caller.py   --proc LIGHTNING \
                        --initials AS \
                        --milestone M23 \
                        --task Overtake_FMD \
                        --bezier \
                        --travelassist-pred \
                        --img-height 100 \
                        --img-width 50 \
                        --algo BC \
                        --training-df-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/processed_M19/processed_AS_M19_Overtake_Performance_2023_05_09-09_56.pkl \
                        --training-image-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/processed_M19/images \
                        --dim-input-feature 6 \
                        --num-epoch 200 \
                        --batch-size 128 \
                        --train-split 0.98 \
                        --val-starting-point 0.5 \
                        --encoder custom \
                        --lr-bc 0.0005 \
                        --activation LeakyReLU \
                        --wandb-entity icc-aitp \
                        --save-model \
                        --model-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/saved_models_BC_M19 \
                        --model-saverate 1 \
                        --num-workers 0 \
                        --dim-input-feature 6 \
                        --mhsa \
                        --print-flag \
                        --multi-opt

# Expert Data evaluation
python3 prog_caller.py --proc EXPERT \
                       --exec-path /home/shamsal/Downloads/ExecutablePackage/SimPilotExe_V12_0_2_B172_Highway3LanesOneway/SimPilotSimServer \
                       --bezier \
                       --milestone M23 \
                        --task Overtake_FMD \
                        --travelassist-pred \
                       --img-height 100 --img-width 50 \
                       --bev-size 50 \
                       --bevoffsetx 10 \
                       --sumo \
                       --controller TravelAssistUnsafe \
                       --maxstep 10000 \
                       --evaluate \
                       --rawdata-path /home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/expert_metric/ \
                       --track \
                       --LaneIDSensor \
                       --num-eps 20 \
                       --rule-based \
                       --record-data