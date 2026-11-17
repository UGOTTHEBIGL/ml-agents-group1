"""
make a csv file containing the hyperparameters of future planned runs
the hyperparameters would be randomly generated based on some predetermined criteria (figuring these out is probably a challenge)
the configurations in the csv will be used to generate temporary yaml files that are used for one run

"""

import csv
import os
import yaml
import random as rand
import numpy as np
import pandas as pd

from pathlib import Path

OUT_CSV = "aiml1/data_collection_scripts/plans/plan.csv"

flattened = {}

random_functions = { #maybe doesnt need to be random but run tests on well defined combinations of each option
    'trainer_type': lambda: rand.choice(['ppo', 'sac']),
    'batch_size': lambda: rand.choice([32, 48, 64, 80]),
    'buffer_size': lambda: rand.choice([8000, 10000, 12000, 16000]),
    'learning_rate': lambda: round(np.exp(np.random.uniform(np.log(0.0001), np.log(0.01))), 5),
    'beta': lambda: round(rand.uniform(0.001, 0.01),4),
    'epsilon': lambda: round(rand.uniform(0.1, 0.5),3),
    'lambd': lambda: round(rand.uniform(0.9, 0.999),4),
    'num_epoch': lambda: rand.choice([2, 3, 4]),
    'learning_rate_schedule': lambda: rand.choice(['linear', 'exponential']),
    'normalize': lambda: rand.choice([True, False]),
    'hidden_units': lambda: rand.choice([32, 64, 96]),
    'num_layers': lambda: rand.choice([1, 2, 3]),
    'vis_encode_type': lambda: rand.choice(['simple', 'nature_cnn']),
    'gamma': lambda: round(rand.uniform(0.95, 0.999), 4),
    'strength': lambda: round(rand.uniform(0.5, 2.0), 3),
    'keep_checkpoints': lambda: rand.choice([3, 5, 10]),
    'max_steps': lambda: rand.choice([100000, 200000, 300000, 500000]),
    'time_horizon': lambda: rand.choice([64, 128, 256, 512, 1024]),
    'summary_freq': lambda: rand.choice([5000, 10000, 20000])
}


def createCSV(path, runs):
    with open(path, "r") as f:  template = yaml.load(f, Loader=yaml.FullLoader)
    
    flatten(flattened, template)
    plan = pd.DataFrame(columns=list(flattened.keys()))

    new_line = {}
    for i in range(runs):
        for param in plan.columns:
            new_line[param] = random_functions[param]()
        plan.loc[i] = new_line

    with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(plan)

    if not os.path.isdir("aiml1/data_collection_scripts/plans"):
        os.makedirs("aiml1/data_collection_scripts/plans")
        print("Folder created")

    plan.to_csv(OUT_CSV)


def assignParameters(data): #recursively matches dict keys and assigns new value with function, used to create temp yaml file to run from csv
    for key in data:
        if not isinstance(data[key], dict):
            data[key] = random_functions[key]()
        else:
            assignParameters(data[key])


def flatten(flattened, data):
    for key in data:
        if not isinstance(data[key], dict):
            flattened[key] = data[key]
        else:
            flatten(flattened, data[key])


if __name__ == "__main__":
    config_path = "aiml1/configs/3DBall_config_test.yaml"
    createCSV(config_path, 10)






