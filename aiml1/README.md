# aiml1: our data collection
## Setup

Follow the main README to set up the environment and the project, then also install these versions, or every run crashes at the very end still don't know why:
```bash
python -m pip install psutil "torch==2.8.0" "protobuf==3.20.1"
```

In Unity open the project and build the envronment you want to use, for example with 3dBall - open the scene of the game you want, for example -  Assets/ML-Agents/Examples/3DBall/Scenes/3DBall,
click Project Settings -> Player -> Run In Background,
then File -> Build Settings -> your OS, Apple silicon → Add Open Scenes → Build
and save it in aiml1/conifgs

## Configs

Copy an original from config/ppo/ or config/sac/ into aiml1/configs/ and play around with the copy, some things shouldn't be changed, i know we can change the hyperparameters and max_steps for sure, so if you play around and something breaks when you have changed it, please add it in the readme.


## How to start

Start the mlagents environtment and run the script with the config, build and a new run name. Here is the command with my configs and names for reference:
```bash
python aiml1/data_collection_scripts/run_once.py aiml1/configs/3DBall_ppo_conf.yaml aiml1/builds/3DBall.app run_001
```
Arguments: script, config, build, run name. Use a new run name every time or the script won't execute.
It trains, measures time and peak RAM, and adds one row to the csv in aiml1/data and gives the raw result in /result directory.



PS: not sure if git will actually commit my folders that are gitignored so if you don't have them please make the same folder structure in you own pulll
```
aiml1/
├── builds/                   
├── configs/                  
├── data/                     
└── data_collection_scripts/  
```
