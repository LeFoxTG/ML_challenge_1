import numpy as np
import torch
from challenge3__2.env import make_env
from challenge3__2.model import AtariActorCritic


# Reuse make_env and AtariActorCritic from Challenge 3
def collect_demonstrations(
    env_id: str,
    checkpoints_path: list,
    n_steps: int = 40_000,
    seed: int = 0,
    device: str = "cpu",
    output_file: str = "demos.npz",
) -> dict:
    """Roll out a saved policy and record ( obs , action ) pairs .
    Returns a dict with keys ’ observations ’ and ’ actions ’,
    each a numpy array of shape ( n_steps , ...) .
    """
    env = make_env(env_id, seed=seed)
    n_actions = env.action_space.n

    model = AtariActorCritic(n_actions).to(device)
    obs_buf, act_buf = [], []
    obs, _ = env.reset()
    episode_rewards = []
    current_episode_reward = 0.0
    for checkpoint_path in checkpoints_path:

        print(f"\nLoading checkpoint: {checkpoint_path}")

        model = AtariActorCritic(n_actions).to(device)

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device
            )
        )

        model.eval()

        obs, _ = env.reset()

        collected = 0

        while collected < n_steps:

            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(obs_t)

            action = logits.argmax(dim=-1).item()

            obs_buf.append(obs)
            act_buf.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            current_episode_reward += reward
            collected += 1

            if terminated or truncated:

                episode_rewards.append(current_episode_reward)

                # print(
                #     f"Episode {len(episode_rewards)} "
                #     f"Reward: {current_episode_reward:.2f}"
                # )

                current_episode_reward = 0.0
                obs, _ = env.reset()

        print(
            f"Collected {n_steps} steps "
            f"from {checkpoint_path}"
        )

    env.close()

    demos = {
        "observations": np.array(obs_buf, dtype=np.float32),
        "actions": np.array(act_buf, dtype=np.int64),
    }

    np.savez_compressed(output_file, **demos)

    print(
        f"\nSaved {len(act_buf)} demonstrations "
        f"to {output_file}"
    )
    
    if len(episode_rewards) > 0:
        rewards = np.array(episode_rewards)
        print("\n===== DEMONSTRATION STATISTICS =====")
        print(f"Episodes    : {len(rewards)}")
        print(f"Mean Reward : {rewards.mean():.2f}")
        print(f"Std Reward  : {rewards.std():.2f}")
        print(f"Max Reward  : {rewards.max():.2f}")
        print(f"Min Reward  : {rewards.min():.2f}")
        print("===================================")

    return demos