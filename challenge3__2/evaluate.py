import argparse
from env import make_env
from ppo_agent import PPOAgent
from gymnasium.wrappers import RecordVideo



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record a video to results/ (implies --no-render)",
    )

    return parser.parse_args()


def evaluate(args):
    render_mode = None
    if args.record:
        render_mode = "rgb_array"
    else:
        render_mode = "human"
    env = make_env("ALE/Pitfall-v5", seed=args.seed, render_mode=render_mode)
    if render_mode == "rgb_array":
        env = RecordVideo(
            env,
            video_folder="Video",
            episode_trigger=lambda ep: True,
            name_prefix="ppo_run"
        )
    
    agent = PPOAgent(env.action_space.n)

    agent.load(args.checkpoint)

    rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total = 0

        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward

        rewards.append(total)
        print(f"[Eval Episode {ep+1}] Reward: {total:.2f}")

    mean_r = sum(rewards) / len(rewards)
    pos = sum(r > 0 for r in rewards) / len(rewards)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Mean Reward: {mean_r:.2f}")
    print(f"Positive Episodes: {pos*100:.2f}%")
    print("==============================")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)