# Challenge 1: Reinforcement Learning (DQN) for Atari - Pitfall

This repository contains the implementation, experiments, and results for training a Deep Q-Network (DQN) agent on the `ALE/Pitfall-v5` environment, as part of Challenge 1 for the Machine Learning course.

## Repository Structure

The repository is structured inside the `challenge1__2` directory as required:
- [`pitfall_dqn.py`](challenge1__2/pitfall_dqn.py): Main script for training, evaluating, and playing the DQN agent.
- [`sweep_configs.json`](challenge1__2/sweep_configs.json): Configuration file containing all inicial hyperparameter experiments, including baseline and OFAT variations.
- [`sweep_phase2.json`](challenge1__2/sweep_phase2.json): Configuration file containing more specific hyperparameters mixing the best results in the first phase.
- [`sweep_curiosity.json`](challenge1__2/sweep_curiosity.json): Configuration file containing hyperparameters for testing agent's behaviour after adding intrinsic curiosity approach.
- [`models/`](challenge1__2/models): Directory where the trained `.zip` models are saved.
- [`logs/`](challenge1__2/logs): TensorBoard event files for all experimental seeds.
- [`challenge1__2_paper.pdf`](challenge1__2/challenge1__2_paper.pdf): IEEE format scientific report detailing our findings.

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

Link to video: 

https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAXIJX_SznqSeJhaJzu1Zluo?e=WN5AEB

In this video we present our work on addressing the sparse reward problem in the Pitfall! environment using Deep Q-Networks (DQN). We conducted a systematic evaluation of different hyperparameter configurations and analyzed their impact on the agent’s performance. After identifying the limitations of standard approaches, we introduced intrinsic curiosity to enhance exploration. Our results show that while hyperparameter tuning provides limited improvements, curiosity plays a key role in enabling the agent to achieve positive rewards. Overall, our work highlights the importance of intrinsic motivation in solving complex reinforcement learning problems with sparse feedback.

### Timestamps

[0:00](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=gDshV3&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7fX0%3D) - Greetings

[0:04](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=jOqfw5&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NC43M319) - The Problem: Sparse Rewards in Atari

[1:20](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=5yubw3&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6ODAuMDZ9fQ%3D%3D) - Deep Q-Networks (DQN)

[2:26](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=NZ1YSg&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MTQ2LjI3fX0%3D) - Experimental Design

[3:19](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=enQ2b3&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MTk5LjY0fX0%3D) - Baseline: The Stay-Still Problem

[3:57](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=CeBRp2&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MjM3LjQ3fX0%3D) - Baseline Tensorboard

[4:43](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=nfO24M&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MjgzLjQxfX0%3D) - Phase 1: OFAT Sweep Results

[5:05](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=47thP2&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzA1LjcxfX0%3D) - Best Phase 1 Model Playing

[5:26](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=6hSrMr&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzI2LjY1fX0%3D) - Phase 2: Combined Configurations

[5:56](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=ooUH0H&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzU2LjA0fX0%3D) - Phase 3: Intrinsic Curiosity Wrapper

[6:30](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=QXEewM&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzkwLjE1fX0%3D) - Best Curiosity Model Playing

[6:50](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=KEaFt7&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NDEwLjQyfX0%3D) - Curiosity Tensorboard

[7:03](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=97e8dH&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NDIzLjI2fX0%3D) - Ablation Study

[8:01](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=ImNQ1w&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NDgxLjM2fX0%3D) - Failure Modes & Key Findings

[8:44](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=kmGCkg&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NTI0LjkzfX0%3D) - Dismissed Technique: Reward Shaping

[9:02](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAU2uGRuxEzoT9jn3fmY5kgo?e=OZapd9&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NTQyLjg0fX0%3D) - Conclusions & Future Work

# Challenge 3: Proximal Policy Optimization for Atari-Pitfall (PPO)
This repository contains the implementation, experiments, and results for training a PPO agent on the `ALE/Pitfall-v5` environment, as part of Challenge 3 for the Machine Learning course.

## Repository Structure

The repository is structured inside the `challenge3__2` directory as required:
- [`train.py`](challenge3__2/train.py): Main script for training the PPO agent.
- [`evaluate.py`](challenge3__2/train.py): Main script for evaluating and see the PPO agent playing.
- [`ppo_agent.py`](challenge3__2/ppo_agent.py): Contains the PPOAgent class, including action selection, trajectory storage, Generalized Advantage Estimation (GAE), PPO updates, checkpoint management, and inference logic.
- [`model.py`](challenge3__2/model.py): Defines the convolutional Actor-Critic neural network architecture used by PPO for Atari observations.
- [`env.py`](challenge3__2/env.py): Creates and configures the ALE/Pitfall-v5 environment with Atari preprocessing, frame stacking, grayscale conversion, resizing, and optional rendering support.
- [`requirements.txt`](challenge3__2/requirements-linux.txt): Lists the Python dependencies required to reproduce the PPO experiments.
- [`checkpoints/`](challenge3__2/checkpoints): Directory where the trained `.pt` chekpoints are saved.
- [`results/`](challenge3__2/results): Graphics of training phase, mean and std of some training configurations.
- [`challenge3__2_paper.pdf`](challenge3__2/challenge3__2_paper.pdf): IEEE format scientific report detailing our findings.

 ## Setup and Installation

To replicate this environment, ensure you have Python 3.11+ installed. Activate your virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the Best Reported Run

After running some baselines and OFAT experiments, our results showed that PPO is better than DQN but not enough for Pitfall's sparse rewards. Our best and stabilized reported run was the baseline for 5,000,000 timesteps or the baseline for 300,000 timesteps, but if we talk about some improvement on exploration the best was the final_baseline for 300,000 timesteps and final baseline for 5,000,000 timesteps , which successfully motivated the agent to explore sub-surface levels.

To exactly reproduce our best training runs (Seed 42), first change the hyperparameters on the [`ppo_agent.py`](challenge3__2/ppo_agent.py) file, for the ones in the [`configs.json`](challenge3__2/configs.json) according to the names of the experiments and execute the respective command:

```bash
python train.py --name baseline_5M --seed 42 --total-steps 5_000_000
python train.py --name baseline_300k --seed 42 --total-steps 300_000
```

### Watching the agent play

To observe the trained agent's behaviour (e.g., descending the stairs, moving around the environments as documented in our IEEE paper), run the next commands, one by one:

```bash
python evaluate.py --checkpoint checkpoints/seed_42/best_baseline_5M.pt
python evaluate.py --checkpoint checkpoints/seed_42/best_baseline_300k.pt
python evaluate.py --checkpoint checkpoints/seed_42/final_baseline_300k.pt
python evaluate.py --checkpoint checkpoints/seed_42/final_baseline_5M.pt
```

## Video

Link to video: 

[https://udistritaleduco-my.sharepoint.com/:v:/g/personal/aaibanezh_udistrital_edu_co/IQAiQLEW8VSCToQ1lvwm7_noAXIJX_SznqSeJhaJzu1Zluo?e=WN5AEB
](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQBrljTc0YgeS50vtCFQJXEHAdgv1LI3XYn8TZ13EYirfic?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=SifWRv)

In this video we present our work on addressing the sparse reward problem in the Pitfall! environment using PPO agen and a comparision with DQN. We conducted a systematic evaluation of different hyperparameter configurations and analyzed their impact on the agent’s performance. Our results show that while hyperparameter tuning provides limited improvements, PPO was more stable but it was not enough to achieve good results.

### Timestamps

[0:00](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQBrljTc0YgeS50vtCFQJXEHAdgv1LI3XYn8TZ13EYirfic?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=r2uE0J) - Greetings and Introduction

[0:41](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQBrljTc0YgeS50vtCFQJXEHAdgv1LI3XYn8TZ13EYirfic?e=UcGRoQ&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NDEuNjd9fQ%3D%3D) - The Problem: Sparse Rewards in Atari

[1:34](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQBrljTc0YgeS50vtCFQJXEHAdgv1LI3XYn8TZ13EYirfic?e=fhGXne&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6OTQuNDV9fQ%3D%3D) - Deep Q-Networks (DQN)

[2:26](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQBrljTc0YgeS50vtCFQJXEHAdgv1LI3XYn8TZ13EYirfic?e=fhGXne&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6OTQuNDV9fQ%3D%3D) - Experimental Design

[3:24](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQBrljTc0YgeS50vtCFQJXEHAdgv1LI3XYn8TZ13EYirfic?e=mKtOku&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MjA0LjI3fX0%3D) - Best PPO Configuration

[5:35](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQBrljTc0YgeS50vtCFQJXEHAdgv1LI3XYn8TZ13EYirfic?e=jZjGUy&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzM1LjMyfX0%3D) - Comparision with DQN


[7:25](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQBrljTc0YgeS50vtCFQJXEHAdgv1LI3XYn8TZ13EYirfic?e=VTyMl5&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NDQ1LjI0fX0%3D) - Conclusions & Future Work

# Challenge 4: Learning from Demonstration with Adversarial Training (GAIL) for Atari - Pitfall
This repository contains the implementation, experiments, and results for training a GAIL agent on the `ALE/Pitfall-v5` environment, as part of Challenge 4 for the Machine Learning course.

## Repository Structure

The repository is structured inside the `challenge4__2` directory as required:
- [`GAIL_train.py`](challenge4__2/GAIL_train.py): Main script for training the GAIL agent using PPO and adversarial imitation learning.
- [`discriminator.py`](challenge4__2/discriminator.py): Defines the GAIL discriminator network used to distinguish expert trajectories from agent-generated trajectories.
- [`bc.py`](challenge4__2/bc.py): Implements Behavior Cloning (BC) training from expert demonstrations.
- [`demonstrations.py`](challenge4__2/demonstrations.py): Generates demonstration datasets by rolling out trained policies and storing observation–action pairs.
- [`evaluate.py`](challenge4__2/evaluate.py): Evaluates trained BC or GAIL policies and renders gameplay episodes.
- [`demos.npz`](challenge4__2/demos.npz): Stored expert demonstrations containing observations and actions used for BC and GAIL training.
- [`BC_Checkpoints/`](challenge4__2/BC_Checkpoints/): Directory containing saved Behavior Cloning checkpoints.
- [`GAIL_Checkpoints/`](challenge4__2/GAIL_Checkpoints/): Directory containing saved GAIL training checkpoints and best-performing policies.
- [`results/`](challenge4__2/results/): Graphics of training phase, mean and std of some training configurations.


 ## Setup and Installation

To replicate this environment, ensure you have Python 3.11+ installed. Activate your virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the Best Reported Run

After running some baselines and OFAT experiments, our results showed that GAIL is better than DQN and PPO but not enough for Pitfall's sparse rewards. Our best and stabilized reported run was the baseline for 2,000,000 timesteps.

To exactly reproduce our best training run (Seed 100) without collecting demos nor training BC again execute the command:

```bash
python GAIL_train.py --steps 2_000_000 --seed 100 --bc_name gail_baseline_5.pt --just_gail
```
If you want to train the BC without collecting demos, execute the command:

```bash
python GAIL_train.py --steps 2_000_000 --seed 100 --bc_name gail_baseline_5.pt --demos_path demos.npz
```

If you want to train the BC and collect the demos (Full workflow), first change the demos_path in [`GAIL_train.py`](challenge4__2/GAIL_train.py) and execute the command:

```bash
python GAIL_train.py --steps 2_000_000 --seed 100 --bc_name gail_baseline_5.pt
```

### Watching the agent play

To observe the trained agent's behaviour (e.g., descending the stairs, moving around the environments as documented in our IEEE paper), run the next commands, one by one:

```bash
python evaluate.py --episodes 5 --checkpoint_path GAIL_Checkpoints/seed_100/best_gail_gail_baseline_5.pt
```

## Video

Link to video: 
{https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQDRtAdmNsh8TKef-eNckKa2AUAqYkIXQKl7fq5sDXWaZFk?](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQDRtAdmNsh8TKef-eNckKa2AUAqYkIXQKl7fq5sDXWaZFk?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=71CoLg)

In this video we present our work on addressing the sparse reward problem in the Pitfall! environment using BC and GAIL agent, and a comparision with DQN and PPO. We conducted a systematic evaluation of different hyperparameter configurations and analyzed their impact on the agent’s performance. Our results show that while hyperparameter tuning provides limited improvements, PPO was more stable but it was not enough to achieve good results.

### Timestamps

[0:00](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQDRtAdmNsh8TKef-eNckKa2AUAqYkIXQKl7fq5sDXWaZFk?e=9C6VPD&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MC41OH19) - Greetings and Introduction

[0:35](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQDRtAdmNsh8TKef-eNckKa2AUAqYkIXQKl7fq5sDXWaZFk?e=0iSB8P&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzQuNjl9fQ%3D%3D) - The Problem: Sparse Rewards in Atari

[1:23](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQDRtAdmNsh8TKef-eNckKa2AUAqYkIXQKl7fq5sDXWaZFk?e=fWDxFJ&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6ODIuOTh9fQ%3D%3D) - What is BC?

[2:26](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQDRtAdmNsh8TKef-eNckKa2AUAqYkIXQKl7fq5sDXWaZFk?e=9YeBUh&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MTM4LjM1fX0%3D) - What is GAIL?

[3:12](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQDRtAdmNsh8TKef-eNckKa2AUAqYkIXQKl7fq5sDXWaZFk?e=Iaj6lX&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MTkyLjg1fX0%3D) - Experimental Design

[4:51](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQDRtAdmNsh8TKef-eNckKa2AUAqYkIXQKl7fq5sDXWaZFk?e=lUxFGD&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MjkxLjY4fX0%3D) - Best GAIL configuration and experiment

[6:22](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQDRtAdmNsh8TKef-eNckKa2AUAqYkIXQKl7fq5sDXWaZFk?e=XqslqF&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6MzgyLjYxfX0%3D) - Comparision with DQN and PPO

[7:26](https://udistritaleduco-my.sharepoint.com/:v:/g/personal/dfarizaa_udistrital_edu_co/IQDRtAdmNsh8TKef-eNckKa2AUAqYkIXQKl7fq5sDXWaZFk?e=CdM5Zh&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifSwicGxheWJhY2tPcHRpb25zIjp7InN0YXJ0VGltZUluU2Vjb25kcyI6NDQ2LjJ9fQ%3D%3D) - Conclusions
