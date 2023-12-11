# SwapTransformer: Highway Overtaking Tactical Planner Model via Imitation Learning on OSHA Dataset 

SwapTransformer investigates the high-level decision-making problem in highway scenarios regarding lane changing and over-taking other slower vehicles. In particular, SwapTransformer aims to improve the Travel Assist feature for automatic overtaking and lane changes on highways. 9 million samples including lane images and other dynamic objects are collected in simulation. This data; Overtaking on Simulated HighwAys (OSHA) dataset is released to tackle this challenge. To solve this problem, an architecture called SwapTransformer is designed and implemented as an imitation learning approach on the OSHA dataset. The problem definition of this research can be summarized in the Figure below:
<br />
<br />
<img src=/images/problem.png width="600" height="220"/>
<br />
![Alt text](/images/problem.png)
<br />
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
Data collection phase is done based on a rule-based driver. The rule-based driver is designed on top of Sumo and Unity engine. For more information about the data collection, please read the paper. 

## 📖 Dataset
Table below shows some details about the dataset collected based on the rule-based driver. Both raw and pre-processed data are mentioned here.
More information about the dataset is availble in the paper and [OSHA Dataset on IEEE Dataport](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs).
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

### 🧠 TRAIN

### 📈 INFERENCE


## 🎥 🚗 Demos


## 🔖 Citation

If you find it useful, please cite our paper as follows:
<br />
```
@article{Title,
  title={SwapTransformer: Highway Overtaking Tactical Planner Model via Imitation Learning on OSHA Dataset },
  author={Shamsoshoara, Alireza and Salihd, Safin and Aghazadeh, Pedram},
  journal={Arxiv},
  pages={XXX YYYY},
  year={2024},
  publisher={Arxiv}
}
```