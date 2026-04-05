# Challenge 1: Reinforcement Learning (DQN) for Atari - Pitfall

This repository contains the implementation, experiments, and results for training a Deep Q-Network (DQN) agent on the `ALE/Pitfall-v5` environment, as part of Challenge 1 for the Machine Learning course.

## Repository Structure

The repository is structured inside the `challenge1__2` directory as required:
- [`pitfall_dqn.py`](challenge1__2/pitfall_dqn.py): Main script for training, evaluating, and playing the DQN agent.
- [`sweep_configs.json`](challenge 1__2/sweep_configs.json): Configuration file containing all inicial hyperparameter experiments, including baseline and OFAT variations.
- [`sweep_phase2.json`](challenge 1__2/sweep_phase2.json): Configuration file containing more specific hyperparameters mixing the best results in the first phase.
- [`sweep_curiosity.json`](challenge 1__2/sweep_curiosity.jwon): Configuration file containing hyperparameters for testing agent's behaviour after adding intrinsic curiosity approach.
- [`models/`](challenge 1__2/models): Directory where the trained `.zip` models are saved.
- [`logs/`](challenge 1__2/logs): TensorBoard event files for all experimental seeds.
- [`challenge1__2_paper.pdf`](challenge 1__2/challenge 1__2_paper.pdf): IEEE format scientific report detailing our findings.

## Setup and Installation

To replicate this environment, ensure you have Python 3.11+ installed. Activate your virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the Best Reported Run

After running 30 baseline and OFAT experiments, our results showed that standard $\epsilon$-greedy exploration is insufficient for Pitfall's sparse rewards. Our best reported run utilised Intrinsic Curiosity `(use_curiosity: true, curiosity_beta: 0.05)` for 500,000 timesteps, which successfully motivated the agent to explore sub-surface levels.

To exactly reproduce our best training run (Seed 42), execute the following command:

```bash
python pitfall_dqn.py --mode train --experiment 17_curiosity_beta005 --model-path models/curiosity/best_model --seed 42 --tensorboard-log logs/curiosity/sweep
```

### Watching the agent play

To observe the trained agent's behaviour (e.g., descending the stairs as documented in our IEEE paper), run the play mode using the generated model:

```bash
python pitfall_dqn.py --mode play --model-path models/curiosity/best_model --episodes 3
```

### Monitoring the Training

To view the learning curves and curiosity-driven exploration metrics, launch TensorBoard:

```bash
python -m tensorboard.main --logdir logs/curiosity/sweep/17_curiosity_beta005 --port 6006
```

Open http://localhost:6006 in your browser to inspect the metrics.

## Video

Link to video: ...

In this video we present our work on addressing the sparse reward problem in the Pitfall! environment using Deep Q-Networks (DQN). We conducted a systematic evaluation of different hyperparameter configurations and analyzed their impact on the agent’s performance. After identifying the limitations of standard approaches, we introduced intrinsic curiosity to enhance exploration. Our results show that while hyperparameter tuning provides limited improvements, curiosity plays a key role in enabling the agent to achieve positive rewards. Overall, our work highlights the importance of intrinsic motivation in solving complex reinforcement learning problems with sparse feedback.
