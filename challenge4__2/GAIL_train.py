import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from challenge3__2.env import make_env
from challenge3__2.model import AtariActorCritic
from challenge4__2.discriminator import GAILDiscriminator
from challenge4__2 import demonstrations
from challenge4__2 import bc

global_seed = 42

def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: list/tensor of rewards from the rollout
        values: list/tensor of value estimates
        dones: list/tensor of done flags
        next_value: bootstrap value from the next state
        gamma: discount factor
        gae_lambda: GAE lambda parameter (between 0 and 1)

    Returns:
        advantages: tensor of advantage estimates
        returns: tensor of returns (advantages + values)
    """
    # Convert to tensors if needed
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float32)
    if not isinstance(dones, torch.Tensor):
        dones = torch.tensor(dones, dtype=torch.float32)

    # Initialize advantage and return buffers
    advantages = []
    gae = 0.0

    # Compute GAE backwards through the trajectory
    values_list = values.tolist() if isinstance(values, torch.Tensor) else values
    next_val = next_value

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = (
                1.0 - dones[t].item()
                if isinstance(dones[t], torch.Tensor)
                else 1.0 - dones[t]
            )
            next_v = next_value
        else:
            next_nonterminal = 1.0 - (
                dones[t].item() if isinstance(dones[t], torch.Tensor) else dones[t]
            )
            next_v = values_list[t + 1]

        delta = rewards[t] + gamma * next_v * next_nonterminal - values_list[t]
        gae = delta + gamma * gae_lambda * next_nonterminal * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values

    return advantages, returns


def train_gail(
    env_id,
    demos_path="demos.npz",
    total_steps=5_000_000,
    horizon=1024,
    n_ppo_epochs=6,
    batch_size=128,
    lr_policy=1e-4,
    lr_disc=3e-4,
    disc_updates_per_rollout=3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=global_seed,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    # --- load demonstrations ---
    data = np.load(demos_path)
    demo_obs = torch.tensor(data["observations"], dtype=torch.float32)
    demo_act = torch.tensor(data["actions"], dtype=torch.long)
    n_demos = len(demo_obs)

    env = make_env(env_id, seed=global_seed)
    n_actions = env.action_space.n

    policy = AtariActorCritic(n_actions).to(device)
    bc_path = (
        f"BC_Checkpoints/"
        f"seed_{seed}/"
        f"{args.bc_name}"
    )
    results_dir = f"results/seed_{args.seed}"
    if os.path.exists(bc_path):
        print(
            f"Loading BC weights: "
            f"{bc_path}"
        )

        policy.load_state_dict(
            torch.load(
                bc_path,
                map_location=device
            )
        )
    disc = GAILDiscriminator(n_actions, use_action=True).to(device)

    opt_policy = optim.Adam(policy.parameters(), lr=lr_policy)
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc)
    bce = torch.nn.BCELoss()

    obs, _ = env.reset()
    ep_return = 0.0
    all_returns = []
    best_return = -float("inf")
    training_steps = []
    training_rewards = []

    for global_step in range(0, total_steps, horizon):


        # ---- rollout collection ----
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf, env_rew_buf = [], [], [], []
        

        for _ in range(horizon):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value = policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            obs_buf.append(obs_t.squeeze(0))
            act_buf.append(action)
            logp_buf.append(dist.log_prob(action))
            val_buf.append(value.squeeze())

            obs, env_reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            done_buf.append(done)
            env_rew_buf.append(env_reward)
            ep_return += env_reward

            if done:
                all_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

        # ---- adversarial reward(replace env reward) ----
        obs_stack = torch.stack(obs_buf).to(device)
        act_one_hot = F.one_hot(torch.stack(act_buf), n_actions).float().to(device)
        act_one_hot = act_one_hot.squeeze(1)
        with torch.no_grad():
            d_scores    = disc(obs_stack, act_one_hot)  # P(expert | s)
            # d_scores = disc(obs_stack) 
        # reward : log D(s , a) -- agent wants to look like the expert
        # adv_rewards = torch.log(d_scores + 1e-8).cpu()
        adv_rewards = -torch.log(1 - d_scores + 1e-8).cpu()  # alternative reward for better stability
        # rew_buf = adv_rewards.tolist()
        rew_buf = (
            adv_rewards
            + 0.05 * torch.tensor(
                env_rew_buf
            )
        ).tolist()

        # ---- GAE advantages ----
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            _, nv = policy(obs_t)
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, nv.item(), gamma, gae_lambda
        )

        # ---- discriminator update ----        

        for _ in range(disc_updates_per_rollout):
            # sample expert mini - batch
            idx_e = torch.randint(0, n_demos, (batch_size,))
            e_obs = demo_obs[idx_e].to(device)
            e_act = demo_act[idx_e].to(device)
            e_act_one_hot = F.one_hot(
                e_act.long(),
                num_classes=n_actions
            ).float().to(device)
            # agent mini - batch
            idx_a = torch.randint(0, horizon, (batch_size,))
            a_obs = obs_stack[idx_a]

            a_act_one_hot = F.one_hot(
                torch.stack(act_buf)[idx_a].long(),
                num_classes=n_actions
            ).float().to(device)
            if a_act_one_hot.dim() == 3:
                a_act_one_hot = a_act_one_hot.squeeze(1)
            d_expert = disc(e_obs, e_act_one_hot)
            d_agent = disc(a_obs, a_act_one_hot)
            d_expert = disc(e_obs, e_act_one_hot)
            d_agent = disc(a_obs, a_act_one_hot)
            loss_disc = bce(d_expert, torch.ones_like(d_expert)) + bce(
                d_agent, torch.zeros_like(d_agent)
            )
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

        # ---- PPO update ----
        act_t = torch.stack(act_buf).to(device)
        logp_t = torch.stack(logp_buf).detach().to(device)
        adv_t = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ret_t = returns.to(device)
        idx = torch.randperm(horizon)
        for _ in range(n_ppo_epochs):
            for start in range(0, horizon, batch_size):
                mb = idx[start : start + batch_size]
                lg, vn = policy(obs_stack[mb])
                dn = Categorical(logits=lg)
                lp_new = dn.log_prob(act_t[mb])
                ent = dn.entropy().mean()
                ratio = (lp_new - logp_t[mb]).exp()

                s1 = ratio * adv_t[mb]
                s2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t[mb]
                l_pi = -torch.min(s1, s2).mean()
                l_vf = ((vn - ret_t[mb]) ** 2).mean()
                loss = l_pi + vf_coef * l_vf - ent_coef * ent

                opt_policy.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt_policy.step()
                
        # print(len(all_returns))
        mean_ret = np.mean(all_returns[-100:])
        std_ret = np.std(all_returns[-100:])
        training_steps.append(global_step)
        training_rewards.append(mean_ret)
        d_acc = (
            (d_expert > 0.5).float().mean() + (d_agent < 0.5).float().mean()
        ) / 2
        print(
            f"step ={global_step} ret ={ mean_ret:.1f} ± {std_ret:.1f} "
            f" disc_loss ={ loss_disc.item():.3f} "
            f" disc_acc ={ d_acc.item():.2f} "
        )
        if not os.path.exists(f"GAIL_Checkpoints/seed_{seed}"):
            os.makedirs(f"GAIL_Checkpoints/seed_{seed}")
        # Save latest checkpoint (policy + discriminator) every rollout
        torch.save(
            policy.state_dict(),
            f"GAIL_Checkpoints/"
            f"seed_{seed}/"
            f"last_gail_{args.bc_name}"
        )
        # print(mean_ret)
        # print(best_return)
        if mean_ret > best_return and global_step >= 100000:  # only consider saving after some initial learning
            best_return = mean_ret
            print(
                f"New best GAIL policy "
                f"(mean return={mean_ret:.1f} ± {std_ret:.1f})"
            )
            # Save best policy and discriminator
            torch.save(
                policy.state_dict(),
                f"GAIL_Checkpoints/"
                f"seed_{seed}/"
                f"best_gail_{args.bc_name}"
            )
        # if global_step % 100_000 == 0:
        #     if not os.path.exists(f"GAIL_Checkpoints/seed_{seed}"):
        #         os.makedirs(f"GAIL_Checkpoints/seed_{seed}")
        #     torch.save(
        #         policy.state_dict(),
        #         f"GAIL_Checkpoints/"
        #         f"seed_{seed}/"
        #         f"gail_{global_step}_{args.bc_name}"
        #     )
    results_path = f"{results_dir}/training_{args.bc_name}.png"
    # ---- Training curve: Reward vs Steps ----
    plt.figure(figsize=(10, 5))

    plt.plot(training_steps, training_rewards)

    plt.xlabel("Training Steps")
    plt.ylabel("Mean Reward (Last 100 Episodes)")
    final_mean = np.mean(all_returns[-100:])
    final_std = np.std(all_returns[-100:])
    plt.title(
    f"GAIL Training Curve\n"
    f"Final Mean Reward: {final_mean:.2f} ± {final_std:.2f}")
    plt.grid(True)

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    results_path = f"{results_dir}/training_{args.bc_name}.png"

    plt.savefig(results_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training plot saved to: {results_path}")
    env.close()
    return policy, disc, all_returns


# Ablation : 5 ,000 demos vs 20 ,000 demos .
# Measure : minimum ’ negative reward episodes ’ per 100 training episodes .


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BC policy on ALE/Pitfall-v5"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2_000_000,
        help="Number of training steps",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=global_seed,
        help="Random seed",
    )
    
    parser.add_argument(
        "--bc_name",
        type=str,
        default="gail_policy.pt",
        help="Name to save the trained GAIL policy",
    )
    
    parser.add_argument(
        "--just_gail",
        action="store_true",
        help="Only train GAIL with a saved BC policy without collecting demos or training BC again",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.just_gail:
        demonstrations.collect_demonstrations(
            env_id="ALE/Pitfall-v5",
            checkpoints_path=[
                "../challenge3__2/checkpoints/seed_42/best_more_lr_more_ent.pt",
                # "../challenge3__2/checkpoints/seed_42/best_baseline_300k.pt",
                "../challenge3__2/checkpoints/seed_42/best_baseline_5M.pt",
                # "../challenge3__2/checkpoints/seed_42/best_more_expl.pt",
                # "../challenge3__2/checkpoints/seed_42/best_more_expl_5M.pt",
            ],
            n_steps=20_000,  # Pitfall demos are short - collect less
            seed=args.seed,
            device="cpu",
            output_file="demos.npz",
        )
        
        bc.train_bc(
            env_id="ALE/Pitfall-v5",
            n_epochs=20,
            batch_size=256,
            lr=1e-4,
                device="cuda" if torch.cuda.is_available() else "cpu",
                output_path=f"BC_Checkpoints/seed_{args.seed}/{args.bc_name}.pt".format(args=args),
        )

    policy, disc, returns = train_gail(
        env_id="ALE/Pitfall-v5",
        total_steps=args.steps,
        horizon=1024,
        disc_updates_per_rollout=10,
        ent_coef=0.02,
        gamma=0.995,
        seed=args.seed,
    )