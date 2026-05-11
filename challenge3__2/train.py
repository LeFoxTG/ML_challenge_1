import argparse
import os
import numpy as np
import time
from env import make_env
from ppo_agent import PPOAgent
from utils import plot_results


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10000)
    parser.add_argument("--name", type=str, default="ppo")


    return parser.parse_args()

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def print_hyperparameters(args, agent):
    learning_rates = [group["lr"] for group in agent.optimizer.param_groups]

    print("\n=== Hiperparametros del entrenamiento ===")
    print(f"total_steps: {args.total_steps}")
    print(f"seed: {args.seed}")
    print(f"log_interval: {args.log_interval}")
    print(f"name: {args.name}")
    print(f"horizon: {agent.horizon}")
    print(f"n_epochs: {agent.n_epochs}")
    print(f"batch_size: {agent.batch_size}")
    print(f"gamma: {agent.gamma}")
    print(f"gae_lambda: {agent.gae_lambda}")
    print(f"clip_eps: {agent.clip_eps}")
    print(f"ent_coef: {agent.ent_coef}")
    print(f"vf_coef: {agent.vf_coef}")
    print(f"learning_rate: {learning_rates[0] if len(learning_rates) == 1 else learning_rates}")
    print(f"device: {agent.device}")

def train(args):
    env = make_env("ALE/Pitfall-v5", seed=args.seed, render_mode=None)
    agent = PPOAgent(env.action_space.n)

    checkpoint_dir = f"checkpoints/seed_{args.seed}"
    results_dir = f"results/seed_{args.seed}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    obs, _ = env.reset(seed=args.seed)

    ep_rewards = []
    ep_reward = 0
    best = -np.inf

    start_time = time.time()
    for step in range(args.total_steps):

        action, logp, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        agent.store(obs, action, logp, reward, done, value)

        obs = next_obs
        ep_reward += reward

        if done:
            ep_rewards.append(ep_reward)
            # print(f"Episode {len(ep_rewards)} finished | Reward: {ep_reward:.2f}")
            ep_reward = 0
            obs, _ = env.reset()

        if len(agent.obs) >= agent.horizon:
            agent.update(obs)

        if step % args.log_interval == 0 and ep_rewards:

            recent_rewards = ep_rewards[-50:]
            mean_r = np.mean(recent_rewards)
            std_r = np.std(recent_rewards)
            min_r = np.min(recent_rewards)
            max_r = np.max(recent_rewards)

            positive_ratio = sum(r > 0 for r in recent_rewards) / len(recent_rewards)

            # ⏱️ Tiempo
            elapsed = time.time() - start_time
            progress = step / args.total_steps
            eta = elapsed * (1 - progress) / (progress + 1e-8)
            print(
                f"[Step {step:>7}] "
                f"Mean: {mean_r:>7.2f} | "
                f"Std: {std_r:>6.2f} | "
                f"Min: {min_r:>6.2f} | "
                f"Max: {max_r:>6.2f} | "
                f"Positive: {positive_ratio*100:>6.2f}% | "
                f"Episodes: {len(ep_rewards)} | "
                f"ETA: {format_time(eta)}"
            )

            # Guardar mejor modelo
            if mean_r > best:
                best = mean_r
                best_checkpoint_path = f"{checkpoint_dir}/best_{args.name}.pt"
                agent.save(best_checkpoint_path)
                print(f"✅ New best model saved: {best_checkpoint_path}")

    final_checkpoint_path = f"{checkpoint_dir}/final_{args.name}.pt"
    results_path = f"{results_dir}/training_{args.name}.png"

    agent.save(final_checkpoint_path)
    plot_results(ep_rewards, results_path)
    print(f"Final model saved: {final_checkpoint_path}")
    print(f"Training results saved: {results_path}")
    print_hyperparameters(args, agent)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
