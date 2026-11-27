"""
make a csv file containing the hyperparameters of future planned runs
the hyperparameters would be randomly generated based on some predetermined criteria (figuring these out is probably a challenge)
the configurations in the csv will be used to generate temporary yaml files that are used for one run each

"""

import re
import os
import yaml
import random as rand
import numpy as np
import pandas as pd
import subprocess
import sys

from pathlib import Path

PLANS = "aiml1/data_collection_scripts/plans"
TEMPORARY_YAML = "aiml1/configs/3DBall_config_temporary.yaml"
TEMPLATE_YAML = "aiml1/configs/3DBall_config_template.yaml"

#lambda functions for hyperparameter selection
random_functions = { 
    'trainer_type': lambda: rand.choice(['ppo']),
    'batch_size': lambda: rand.choice([32, 48, 64, 80]),
    'buffer_size': lambda: rand.choice([8000, 10000, 12000, 16000]),
    'learning_rate': lambda: round(np.exp(np.random.uniform(np.log(0.0001), np.log(0.01))), 5),
    'beta': lambda: round(rand.uniform(0.001, 0.01),4),
    'epsilon': lambda: round(rand.uniform(0.1, 0.5),3),
    'lambd': lambda: round(rand.uniform(0.9, 0.999),4),
    'num_epoch': lambda: rand.choice([2, 3, 4]),
    'learning_rate_schedule': lambda: rand.choice(['linear']), #had exponential but does not work
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

with open(TEMPLATE_YAML, "r") as f: template = yaml.load(f, Loader=yaml.FullLoader)

#creates csv file where each row holds hyperparameters for one run
def createCSV(runs=10):
    flattened = {}
    copy = template
    flatten(flattened, copy)
    plan = pd.DataFrame(columns=list(flattened.keys()))

    new_line = {}
    for i in range(runs):
        for param in plan.columns:
            new_line[param] = random_functions[param]()
        plan.loc[i] = new_line

    with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(plan)

    #adds new plan to plans folder
    next_plan_id = 1
    if not os.path.isdir(PLANS):
        os.makedirs(PLANS)
        print("Folder created")
    else:
        pattern = re.compile(r"^plan(\d+).csv$")
        for entry in os.listdir(PLANS):
            match = pattern.match(entry)
            if match:
                plan_id = int(match.group(1))
                if plan_id >= next_plan_id:
                    next_plan_id = plan_id + 1

    plan.to_csv(PLANS+'/plan'+str(next_plan_id)+'.csv')

#flattens yaml sourced dictionary for easy use
def flatten(flattened : dict, config_template : dict):
    for key in config_template:
        if not isinstance(config_template[key], dict):
            flattened[key] = config_template[key]
        else:
            flatten(flattened, config_template[key])

#creates dictionary that matches shape of template
def createTempDict(config_template : dict, plan : pd.Series):
    for key in config_template:
        if not isinstance(config_template[key], dict):
            config_template[key] = plan[key]
        else:
            createTempDict(config_template[key], plan)

#calls run_once.py for each plan entry. calls entire thing each time because idk how to reset enviroment
def runPlan(plan_num=1):
    plan = pd.read_csv(PLANS+"/plan"+str(plan_num)+".csv")
    temp = template
    for run, plan_row in plan.iterrows():
        createTempDict(temp, plan_row)
        with open(TEMPORARY_YAML, 'w') as outfile: yaml.dump(temp, outfile, default_flow_style=False, sort_keys=False)
        # track the result of the training, success or fail
        result = subprocess.run([sys.executable, "aiml1/data_collection_scripts/run_once.py", TEMPORARY_YAML, "aiml1/builds/3DBall.exe", "plan_" + str(plan_num) + "_run_" + str(run)])
        if result.returncode != 0: # anything other than 0 means training failed
            print("plan_" + str(plan_num) + "_run_" + str(run) + " failed") # shows you which


#python aiml1\data_collection_scripts\manage_plan.py to run whatever is in here
if __name__ == "__main__":
    #createCSV()
    runPlan()