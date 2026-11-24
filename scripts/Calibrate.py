import argparse
import sys 
import numpy as np
from pathlib import Path

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


def random_actions(specs, n_agents, rng):
    continuous = rng.uniform(-1.0, 1.0, size=(n_agents, specs.continuous_size))
    discrete = np.zeros((n_agents, specs.discrete_size), dtype=np.int32)
    for i, branch_size in enumerate(specs.discrete_branches):
        discrete[:, i] = rng.integers(0, branch_size, size=n_agents)
    return ActionTuple(continuous=continuous, discrete=discrete)

def random_steps(env, behavior, n_steps, rng, n_episodes):
    specs = env.behavior_specs[behavior].action_spec
    n_terminal_total = 0
    episode_rewards = {}
    completed_episodes = []
    episode_counts = {}
    n_interrupted = 0
    initial_decision, initial_terminal = env.get_steps(behavior)
    agent_ids = list(initial_decision.agent_id)
    n_agents = len(agent_ids)
    print(f"Agents found: {n_agents}")
    print(f"Agent IDs: {agent_ids}")
    if n_agents == 0:
        raise ValueError("No agents found")

    base = n_episodes // n_agents
    remainder = n_episodes % n_agents

    episode_targets = {}

    for i, agent_id in enumerate(agent_ids):
        episode_targets[agent_id] = base + (1 if i < remainder else 0)

    for t in range(n_steps):
        decision, terminal = env.get_steps(behavior)
        for agent_id, reward in zip(terminal.agent_id, terminal.reward):
            if agent_id not in episode_targets:
                raise RuntimeError(f"Unexpected agent ID: {agent_id}")
            total_reward = episode_rewards.get(agent_id, 0.0) + reward

            if episode_counts.get(agent_id, 0) < episode_targets[agent_id]:
                n_interrupted += np.count_nonzero(terminal.interrupted)
                completed_episodes.append(total_reward)
                episode_counts[agent_id] = episode_counts.get(agent_id, 0) + 1

            episode_rewards[agent_id] = 0.0

        for agent_id, reward in zip(decision.agent_id, decision.reward):
            episode_rewards[agent_id] = (
                episode_rewards.get(agent_id, 0.0) + reward
            )

        n_terminal_total += len(terminal)

        if len(completed_episodes) >= n_episodes:
            break
    
        if t < 5:
            print(t, "Decision: ", len(decision), "Terminal: ", len(terminal),
                  "Reward sum: ", decision.reward.sum() + terminal.reward.sum())
        if len(decision) > 0:
            env.set_actions(behavior, random_actions(specs, len(decision), rng))
        env.step()
    print(f"Terminal steps seen in {t + 1} steps: {n_terminal_total}")
    print(f"Interrupted episodes: {n_interrupted}")
    return completed_episodes

def main():
    # parsing settings (define arguments)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Path to the unity build (e.g. builds/3DBall/3DBall.exe)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=5200)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100000)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be greater than 0")

    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than 0")

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
        rewards = random_steps(env, names[0], args.max_steps, rng, args.episodes)

        if len(rewards) < args.episodes:
            raise RuntimeError(
                f"Only {len(rewards)} of {args.episodes} episodes completed"
       )


        rewards = np.array(rewards)
        n = len(rewards)
        mean = rewards.mean()
        std = rewards.std(ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n)
        behavior = names[0].split("?")[0]

        print(f"Behaviour: {behavior} seed: {args.seed} episodes: {n}")
        print(f"Episode reward: mean {mean:.4f} std {std:.4f} sem {sem:.4f} "
              f"min {rewards.min():.4f} max {rewards.max():.4f}")
        print(f"R_random: {mean:.4f}")

        print(f"{behavior}:\n random_reward: {mean:.4f}")

    
    finally:
        env.close()

if __name__== "__main__":
    main()