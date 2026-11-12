import argparse
import sys 
import numpy as np
from pathlib import Path

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


def random_actions(specs, n_agents, rng):
    continuous = rng.uniform(-1.0, 1.0, size=(n_agents, specs.continuous_size))
    discrete = np.zeros((n_agents, specs.discrete_size), dtype=np.int32)
    #TODO implement for discrete games
    return ActionTuple(continuous=continuous, discrete=discrete)

def random_steps(env, behavior, n_steps, rng):
    specs = env.behavior_specs[behavior].action_spec
    n_terminal_total = 0
    for t in range(n_steps):
        decision, terminal = env.get_steps(behavior)
        n_terminal_total += len(terminal)
        if t < 5 or len(terminal) > 0:
            print(t, "Decision: ", len(decision), "Terminal: ", len(terminal),
                  "Reward sum: ", decision.reward.sum() + terminal.reward.sum())
        if len(decision) > 0:
            env.set_actions(behavior, random_actions(specs, len(decision), rng))
        env.step()
    print(f"Terminal steps seen in {n_steps} steps: {n_terminal_total}")

def main():
    # parsing settings (define arguments)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Path to the unity build (e.g. builds/3DBall/3DBall.exe)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=5200)
    args = parser.parse_args()

    if not Path(args.env).exists():
        sys.exit(f"Build not found: {args.env}")

    env = UnityEnvironment(
        file_name = args.env,
        no_graphics = True,
        seed = args.seed,
        base_port = args.base_port,
        timeout_wait = 120,
    )
    try: 
        env.reset()
        names = list(env.behavior_specs.keys())
        print(f"Behaviours: {names}")
        if len(names) != 1:
            sys.exit(f"Expected exactly 1 behaviour, but found {len(names)}")

        rng = np.random.default_rng(args.seed)
        random_steps(env, names[0], 300, rng)

    
    finally:
        env.close()

if __name__== "__main__":
    main()