import sys
import os
sys.path.append(os.path.abspath(".."))

import argparse
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np

from challenge3__2.env import make_env
from challenge3__2.model import AtariActorCritic


def evaluate(
    env_id="ALE/Pitfall-v5",
    checkpoint_path="bc_policy.pt",
    episodes=5,
    mode="human",
    device="cpu",
):
    device = torch.device(device)
    mode = (
        "rgb_array"
        if args.mode == "record"
        else "human"
    )
    env = make_env(
        env_id,
        render_mode=mode,
    )

    if mode == "rgb_array":
        env = RecordVideo(
            env,
            video_folder="Video",
            episode_trigger=lambda ep: True,
            name_prefix="gail_run"
        )
    
    n_actions = env.action_space.n

    model = AtariActorCritic(n_actions).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

    model.eval()

    rewards = []

    print(f"\nLoaded model: {checkpoint_path}")
    print(f"Evaluating for {episodes} episodes...\n")

    for ep in range(episodes):

        obs, _ = env.reset()

        done = False
        truncated = False

        total_reward = 0
        steps = 0

        while not (done or truncated):

            obs_t = torch.tensor(
                np.array(obs),
                dtype=torch.float32,
                device=device,
            )

            if obs_t.shape[-1] == 4:
                obs_t = obs_t.permute(2, 0, 1)

            obs_t = obs_t.unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(obs_t)
                action = torch.argmax(logits, dim=-1).item()

            obs, reward, done, truncated, _ = env.step(action)

            total_reward += reward
            steps += 1

        rewards.append(total_reward)

        print(
            f"[Episode {ep + 1}/{episodes}] "
            f"Reward: {total_reward:.2f} | "
            f"Steps: {steps}"
        )

    env.close()

    rewards = np.array(rewards)

    print("\n===== FINAL RESULTS =====")
    print(f"Mean Reward: {rewards.mean():.2f}")
    print(f"Std Reward : {rewards.std():.2f}")
    print(f"Max Reward : {rewards.max():.2f}")
    print(f"Min Reward : {rewards.min():.2f}")
    print("=========================")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BC/PPO policy on ALE/Pitfall-v5"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="BC_Checkpoints/bc_policy.pt",
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        help="Render mode (human, rgb_array)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate(
        checkpoint_path=args.checkpoint_path,
        episodes=args.episodes,
        mode=args.mode,
    )