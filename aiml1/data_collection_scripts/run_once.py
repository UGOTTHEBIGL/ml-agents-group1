import csv
import os
import platform
import subprocess
import sys
import time
import psutil
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REWARD_TAG = "Environment/Cumulative Reward"
OUT_CSV = "aiml1/data/runs.csv"


def main():
    config_file = sys.argv[1]
    env_file = sys.argv[2]
    run_id = sys.argv[3]

    if os.path.exists("results/" + run_id):
        sys.exit("run_id " + run_id + " already used")

    # command to run mlagents-learn
    cmd = ["mlagents-learn", config_file, "--env=" + env_file, "--run-id=" + run_id, "--no-graphics"]
    start_time = time.time()
    # starting a child proccess directly from python script, so we can monitor the ram
    process = subprocess.Popen(cmd)

    # keep checking RAM - keeps only highet value
    peak_ram = 0
    while process.poll() is None:
        try:
            # we get our pid of the main proccess and then we get its children recursively, we have to do this
            # in order to get the full memory that is being used, because our main proccess
            # is the mlagents-learn which spawns a worker proccess which then spawns unity, so we need to get all of them and sum their memory usage

            main_proc = psutil.Process(process.pid) # mlagents-learn
            all_procs = [main_proc] + main_proc.children(recursive=True) # worker + unity

            current_ram = 0
            for p in all_procs:
                current_ram += p.memory_info().rss # we choose rss because it is the actual memory used by the process not the VM

            if current_ram > peak_ram:
                peak_ram = current_ram
        except psutil.NoSuchProcess: # if the process is already dead, we just ignore it and continue
            pass

        time.sleep(1)

    seconds = time.time() - start_time

    if process.returncode == 0:
        status = "completed"
    else:
        status = "crashed"

    # read the config to collect the data
    saved_config_path = "results/" + run_id + "/configuration.yaml"
    if os.path.exists(saved_config_path):
        with open(saved_config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

    # get the first behavior name and its settings and put them in dictionary
    behaviors = config["behaviors"]
    game_name = list(behaviors)[0]
    settings = behaviors[game_name]


    hp = settings.get("hyperparameters", {})
    net = settings.get("network_settings", {})

    # final reward = average of the last 10%
    final_reward = None
    reward_log = "results/" + run_id + "/" + game_name

    if os.path.exists(reward_log):
        # open the tensorboard log and get the reward values
        log = EventAccumulator(reward_log)
        log.Reload()

        if REWARD_TAG in log.Tags()["scalars"]:
            rewards = [point.value for point in log.Scalars(REWARD_TAG)]

            # how many values make up the last 10% (at least 1)
            how_many = max(1, len(rewards) // 10)

            # gets the last 10% of values
            last_rewards = rewards[-how_many:]
            final_reward = round(sum(last_rewards) / len(last_rewards), 3)

    # hardware info
    cpu_name = platform.processor()
    if platform.system() == "Darwin":  # needed for Mac because we only get "arm" on Mac
        res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        cpu_name = res.stdout.strip()

    # this calculation is just so we can see everything in GiB because that's is the usual standart
    total_ram_gib = round(psutil.virtual_memory().total / (1024 * 1024 * 1024), 1)
    peak_ram_mib = round(peak_ram / (1024 * 1024), 1)

    # build the row for the CSV
    row = {
        "run_id": run_id,
        "status": status,
        "env_name": game_name,
        "algorithm": settings.get("trainer_type"),
        "os": platform.system(),
        "cpu": cpu_name,
        "cores": psutil.cpu_count(),
        "ram_total_gib": total_ram_gib,
        "learning_rate": hp.get("learning_rate"),
        "batch_size": hp.get("batch_size"),
        "buffer_size": hp.get("buffer_size"),
        "beta": hp.get("beta"),
        "epsilon": hp.get("epsilon"),
        "lambd": hp.get("lambd"),
        "num_epoch": hp.get("num_epoch"),
        "tau": hp.get("tau"),
        "steps_per_update": hp.get("steps_per_update"),
        "init_entcoef": hp.get("init_entcoef"),
        "hidden_units": net.get("hidden_units"),
        "num_layers": net.get("num_layers"),
        "max_steps": settings.get("max_steps"),
        "summary_freq": settings.get("summary_freq"),
        "time_s": round(seconds, 1),
        "ram_peak_mib": peak_ram_mib,
        "final_reward": final_reward,
    }

    # save the row
    folder_path = os.path.dirname(OUT_CSV)
    os.makedirs(folder_path, exist_ok=True)

    file_exists = os.path.exists(OUT_CSV)

    # check for different columns if we are adding rows to an existing CSV
    if file_exists:
        with open(OUT_CSV, "r", newline="") as f:
            header = next(csv.reader(f), [])
        if header != list(row.keys()):
            sys.exit(OUT_CSV + " has different columns")

    with open(OUT_CSV, "a", newline="") as f:
        columns = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=columns)

        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(row)


if __name__ == "__main__":
    main()
