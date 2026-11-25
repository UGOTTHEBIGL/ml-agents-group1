import subprocess
import sys
import time
from datetime import datetime

"""
Runs run_once.py multiple times.
How to use:
----------
python aiml1/data_collection_scripts/batch_runner.py <config> <env> <prefix> [start_num] [count]

Example - run 10 tests starting from test_001:
python aiml1/data_collection_scripts/batch_runner.py aiml1/configs/3DBall_ppo.yaml aiml1/builds/3DBall.exe test 1 10

Example - run 5 more runs starting from test_011:
python aiml1/data_collection_scripts/batch_runner.py aiml1/configs/3DBall_ppo.yaml aiml1/builds/3DBall.exe test 11 5

If a test id is taken, it will jump to the next automatically.
You can stop anytime with Ctrl+C. Already finished runs will stay logged in the CSV.
"""

def run_one(config, env, run_id):
    cmd = [
        "python", "aiml1/data_collection_scripts/run_once.py",
        config,
        env,
        run_id
    ]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting: {run_id}")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"✓ Finished successfully: {run_id}")
        else:
            print(f"✗ Failed: {run_id}")
    except KeyboardInterrupt:
        print(f"\nStopped by user during: {run_id}")
        raise


def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python batch_runner.py <config> <env> <prefix> [start_num] [count]")
        print("\nExample:")
        print("  python batch_runner.py aiml1/configs/3DBall_ppo.yaml aiml1/builds/3DBall.exe ball 1 5")
        sys.exit(1)

    config = sys.argv[1]
    env = sys.argv[2]
    prefix = sys.argv[3]
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    count = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    print(f"Will run {count} jobs starting from {prefix}_{start:03d}")
    print("Press Ctrl+C anytime to stop. Finished runs stay in the CSV.\n")

    try:
        for i in range(start, start + count):
            run_id = f"{prefix}_{i:03d}"
            run_one(config, env, run_id)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nBatch stopped by user.")
        print("All finished runs are saved in aiml1/data/runs.csv")


if __name__ == "__main__":
    main()