BEST PPO RUN
```bash
python evaluate.py --checkpoint checkpoints/seed_42/best_baseline_300k.pt
```
or
```bash
python evaluate.py --checkpoint checkpoints/seed_42/best_baseline_5M.pt
```

SEEDS
- 42
- 100
- 2026

POINTERS
- PPO results
    - [`challenge3__2/results/`](../challenge3__2/results/) 
- PPO Checkpoints
    - [`challenge3__2/checkpoints/`](../challenge3__2/checkpoints/)
- DQN logs
    - [`challenge1__2/logs/`](../challenge1__2/logs/)


SUMMARY

The experiments showed a clear difference between Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO) in how they performed in the Pitfall! game, where the rewards were rare. DQN adapted more quickly during the early training because of its value-based updates and epsilon-greedy exploration. This allowed the agent to sometimes discover new states and show stronger exploratory behavior, especially when combined with intrinsic curiosity mechanisms. However, this intense learning process also led to unstable reward evolution and frequent convergence toward local optima that were not very good. In contrast, PPO had much smoother and more stable training dynamics because of its clipped policy optimization objective and actor-critic architecture. The reward oscillations were smaller, the convergence was more gradual, and policies evolved more conservatively across training. However, this increased stability reduced the agent's ability to escape suboptimal behaviors in a setting where the rewards are sparse. PPO agents usually did the same things over and over, like standing still or jumping in place. DQN agents, on the other hand, sometimes did different things. In practice, the results suggest that DQN preferred to explore, even if it meant sacrificing stability. On the other hand, PPO focused on stability in the short term but had trouble effectively exploring over long periods.
