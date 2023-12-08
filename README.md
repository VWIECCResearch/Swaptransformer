# SwapTransformer: Highway Overtaking Tactical Planner Model via Imitation Learning on OSHA Dataset 

SwapTransformer investigates the high-level decision-making problem in highway scenarios regarding lane changing and over-taking other slower vehicles. In particular, SwapTransformer aims to improve the Travel Assist feature for automatic overtaking and lane changes on highways. 9 million samples including lane images and other dynamic objects are collected in simulation. This data; Overtaking on Simulated HighwAys (OSHA) dataset is released to tackle this challenge. To solve this problem, an architecture called SwapTransformer is designed and implemented as an imitation learning approach on the OSHA dataset. The problem definition of this research can be summarized in the Figure below:
<br />
<br />
<img src=/images/problem.png width="600" height="220"/>
<!--- ![Alt text](/images/problem.PNG) --->
<br />
<br />

## Data Collection

## Dataset

## Running SwapTransformer

## Demos

### EXPERT (Sumo)
```
python prog_caller.py --initials <YOUR_INITIALS> --milestone <MILESTONE> --task <TASK_DEF> --proc EXPERT --record-data --speed-step 1 --steer-step 5 --num-eps 5 --virtual-speedlimit 15 --target-speed 14 --bevoffsetx 0 --maxsteps 6000 --rawdata-path <RAW_PATH> --controller SumoController
```

### EXPERT (Human)
```
python prog_caller.py --initials <YOUR_INITIALS> --milestone <MILESTONE> --task <TASK_DEF> --proc EXPERT --record-data --speed-step 1 --steer-step 5 --num-eps 5 --virtual-speedlimit 15 --target-speed 14 --bevoffsetx 0 --maxsteps 6000 --rawdata-path <RAW_PATH> --human
```


## PREPROCESS
```
python prog_caller.py --initials <YOUR_INITIALS> --milestone <MILESTONE> --task <TASK_DEF> --proc PREPROCESS --rawdata-path <RAW_PATH> --processeddata-path <PROCESSED_PATH>
```

## TRAIN
```
python3 prog_caller.py --proc TRAIN --training-df-path <PATH>  --training-image-path <PATH> --validation-df-path <PATH> --validation-image-path <PATH> --img-height 100 --img-width 50 --algo BC --bezier --travelassist-pred --base-model transformer --batch-size 256 --residual --lr-bc 0.0001 --car-network --encoder resnet18
```

## INFERENCE
```
python3 prog_caller.py --proc INFERENCE --infer-type Online --exec-path <PATH-TO-EXEC> --mhsa --bezier --travelassist-pred --algo BC --model-path <PATH-TO-MODEL> --model-name <MODEL-NAME> --img-height 100 --img-width 50 --sumo --controller TravelAssistUnsafe --maxstep 10000 --track
```

## Randomization
To control and allow randomization of each vtype, we have a copy of the original in the scenedata folder, and will modify the vTypes.add.xml file in the simulation directory. Make sure to have each vType available in your projectsetting.json file, they are as follows:    
"VTypes": [
      "vTypeBase",
      "vTypeFast",
      "vTypeSlow",
      "vTypeAgressive",
      "vTypeImperfect"
    ],
## SWEEP
```
wandb sweep --project <PROJECT_NAME> --entity <ENTITY_NAME> sweep.yaml
<CUDA_VISIBLE_DEVICE> wandb agent <SWEEP_ID> <NUM_OF_RUNS>
```
INFO FOR THE REPO

    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "prog_caller.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": [
                // *********************** General ***********************
                // "--initials",
                // "INIT",
                // "--proc",
                // "EXPERT",
                // "| EXPERT | PREPROCESS | MERGE | TRAIN | INFERENCE",
                // "--milestone",
                // "M11",
                // "--task",
                // "SpeedMatch",
                // "--visu",
                // "--print-flag",
                // "--curve-turn",
                // "--target-radius",
                // "30",
                // "--vis-rate",
                // "60",
                // "--print-rate",
                // "30",

                // *********************** Simulation ***********************
                // "--scene",
                // "OSCOVAL",
                // "EHRA | MEGA | BERLIN | OSCOVAL (default)",
                // "--no-graphic",
                // "--editor",
                // "--sim-steptime",
                // "20",
                // "--img-height",
                // "100",
                // "--img-width",
                // "60",
                // "--bev-size",
                // "60",
                // "--controller",
                // "ExternalEgoCarController",
                // "SumoController" | "ExternalEgoCarController",
                // "--exec-path",
                // "/home/shamsal/Downloads/ExecutablePackage/SimPilotExe_V9_1_3_B109_OSCOval/SimPilotSimServer",
                // "--num-eps",
                // "5",
                // "--bevoffsetx",
                // "0",
                // "--bevoffsety",
                // "0",
                // "--maxsteps",
                // "6000",


                // *********************** Approach ***********************
                // "--pose-steptime",
                // "500",
                // "--num-poses",
                // "5",
                // "--num-framestack",
                // "5",
                // "--num-featurespose",
                // "3",

                // *********************** Data collection ***********************
                // "--human",
                // "--record-data",
                // "--target-speed",
                // "14",
                // "--curve-turn",
                // "--target-radius",
                // "30",
                // "--speed-step",
                // "1",
                // "--steer-step",
                // "5",
                // "--virtual-speedlimit",
                // "15",

                // *********************** Pre-process ***********************
                "--rawdata-path",
                "/home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/raw/",
                "--processeddata-path",
                "/home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/processed",
                // "--dest",
                // "Apollo",
                // "--large-df-path",
                // "Apollo/aitp/large_df",
                // "--img-folder",
                // "Apollo/aitp/image",
                // "--archive-path",
                // "Apollo/aitp/Archive",
                // "--compress",
                // "--compress-name",
                // "compressed_test.tar.gz",

                // *********************** Training and Inference ***********************
                // "--algo",
                // "GAN",
                // "GAN | GAIL | BC",
                
                // "--training-df-path",
                // "/home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/processed/processed_2023_02_03-16_02.pkl",
                // "--training-image-path",
                // "/home/shamsal/Downloads/cpc_aitp_simpilot_eval/data/training_data/processed/images",
                // "--dim-input-feature",
                // "5",
                // "--num-epoch",
                // "200",
                // "--model-path",
                // "Model path for the inference demonstrations",
                // "--disc-freq",
                // "4",
                // "--save-model",
                // "--model-saverate",
                // "20",
                // "--replay-data",
                // "--model-name",
                // "checkpoint-2023_01_19-15_30_update_99.pth",

                // *********************** Monitoring and WANDB ***********************
                // "--track",
                // "--wandb-name",
                // "GAN",
                // "--wandb-entity",
                // "icc-aitp",

            ]
        }
    ]