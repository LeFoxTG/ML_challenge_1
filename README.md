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
