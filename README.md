# SwapTransformer: Highway Overtaking Tactical Planner Model via Imitation Learning on OSHA Dataset 

SwapTransformer investigates the high-level decision-making problem in highway scenarios regarding lane changing and over-taking other slower vehicles. In particular, SwapTransformer aims to improve the Travel Assist feature for automatic overtaking and lane changes on highways. 9 million samples including lane images and other dynamic objects are collected in simulation. This data; Overtaking on Simulated HighwAys (OSHA) dataset is released to tackle this challenge. To solve this problem, an architecture called SwapTransformer is designed and implemented as an imitation learning approach on the OSHA dataset. The problem definition of this research can be summarized in the Figure below:
<br />
<br />
<img src=/images/problem.png width="600" height="220"/>
<br />
<!-- ![Alt text](/images/problem.png) -->
<br />
<br />

SwapTransformer architecture is demonstrated in Figure below. This architecture includes main tasks and auxiliary tasks. Those main tasks (lane change action and ego speed) directly interact with the travel-assist controller. Those auxiliary tasks including future trajectory estimation and the CarNetwork matrix are used as benefits for the model to better understand the agents' interactions and future decision-making. The swapping feature for the core part of the model is explained in the paper in more detail.

<br />
<!-- <img src=/images/approach_01.png width="400" height="320"/> -->

![Alt text](/images/approach_01.png)
<br />


## 🛠️ Requirements
To run different parts of this repo, there is a requirement list for the Python packages which are included in the requirement.txt file. Keep in mind that all packages are tested on Python 3.8.0.
To install all packages in your conda environment, simply create a new environment and install the packages.

```sh
conda create --name env_swaptransformer python==3.8.0 -y
conda activate env_swaptransformer
pip install -r requirement.txt
```

## ⌛ Data Collection
The data collection phase is done based on a rule-based driver. The rule-based driver is designed on top of the Sumo and Unity engine. For more information about the data collection, please read the paper.

![Alt Text](https://github.com/VWIECCResearch/Swaptransformer/blob/main/images/datacollection.gif)

## 📖 Dataset
The table below shows some details about the dataset collected based on the rule-based driver. Both raw and pre-processed data are mentioned here.
More information about the dataset is available in the paper and [OSHA Dataset on IEEE Dataport](https://ieee-dataport.org/open-access/LINK_GOES_HERE).
<!-- <br />
<br />
<img src=/images/dataset.png width="400" height="320"/> -->
<!-- <br /> -->
<br />

|  Description | Raw Data | Processed Data|
| ------------ | ------------ | ------------ |
| Number of pickle files | 900 | 1 |
| Pickle file size (single) | 34.1 MB | 61 GB |
| Image size | 5.7 MB (episode) | 35 GB |
| Total number of samples | 8,970,692 | 8,206,442 |
| Lane change commands | 5,147 | 69,119 |
| Left lane commands | 2,648 | 35,859 |
| Right lane commands | 2,499 | 33,260 |
| Transition commands | 0 | 1,468,115 |
| Number of episodes | 900 | 834 |
| Samples per episode | 10,000 | 9,839 (Average) |
| Speed limit values | {30, 40, ..., 80} (km/h) | {30, 40, ..., 80} (km/h) |
| Ego speed range | [0, 79.92] (km/h) | [0, 79.92] (km/h) |

<br />


## Running SwapTransformer


### 🧮 PREPROCESS
[OSHA Dataset on IEEE Dataport](https://ieee-dataport.org/open-access/LINK_GOES_HERE) shares more information about the pre-processing phase and how raw data is different than the pre-processed data. The appendix section in the paper also gives more information for pre-processing.


### 🧠 TRAIN


### 📈 INFERENCE
To evaluate inference, different baselines and the proposed approach were run on 50 different episodes for comparison. These 50 episodes of testing and inference have different traffic behavior. The table below shows some of the results:

<br />
<!-- <img src=/images/inference_table.png width="580" height="200"/> -->

![Alt text](/images/inference_table.png)
<br />

<!-- |  Metrics | 1) Speed difference (m/s) ↓  | 2) Time to finish (s) ↓ | 3) Left overtake ratio ↑ | 
| ------------ | ------------ | ------------ | ------------ |
| Traffic | Low  Med High | Low Med High | Low Med High |
|  | 3.19 ± 0.7 4.16 ± 0.98 4.37 ± 0.77  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  | -->

## 🎥 🚗 Demos
The grid below shows how the simulation looks like when the SwapTransformer controls the ego vehicle. Those future pose estimations are shown in each image.
<br />
<!-- <img src=/images/inference_grid_01.png width="400" height="320"/> -->

![Alt text](/images/inference_grid_01.png)
<br />


## 🔖 Citation

If you find this work useful, please cite our paper as follows:
<br />
```
@article{SwapTransformer2024,
  title={SwapTransformer: Highway Overtaking Tactical Planner Model via Imitation Learning on OSHA Dataset },
  author={Shamsoshoara, Alireza and Salih, Safin and Aghazadeh, Pedram},
  journal={Arxiv},
  pages={XXX YYYY},
  year={2024},
  publisher={Arxiv}
}
```