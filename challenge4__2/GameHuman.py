import numpy as np
import pygame
import torch
import sys
import os

sys.path.append(os.path.abspath(".."))
from challenge3__2.env import make_env

# =========================
# KEYBOARD MAPPING
# =========================

KEY_TO_ACTION = {
    pygame.K_LEFT: 4,
    pygame.K_RIGHT: 3,
    pygame.K_SPACE: 2,
    pygame.K_UP: 1,
}

# ALE/Pitfall actions may differ.
# Print env.unwrapped.get_action_meanings()
# to verify.



def build_key_to_action(env):
    meanings = env.unwrapped.get_action_meanings()
    mapping = {}

    # Ejemplo de mapeo básico
    if "LEFT" in meanings:
        mapping[pygame.K_LEFT] = meanings.index("LEFT")
    if "RIGHT" in meanings:
        mapping[pygame.K_RIGHT] = meanings.index("RIGHT")
    if "UP" in meanings:
        mapping[pygame.K_UP] = meanings.index("UP")
    if "FIRE" in meanings:
        mapping[pygame.K_SPACE] = meanings.index("FIRE")

    return mapping

def get_action(env):
    action = 0  # NOOP por defecto

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        action = env.unwrapped.get_action_meanings().index("LEFTFIRE")
    elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        action = env.unwrapped.get_action_meanings().index("RIGHTFIRE")
    elif keys[pygame.K_LEFT]:
        action = env.unwrapped.get_action_meanings().index("LEFT")
    elif keys[pygame.K_RIGHT]:
        action = env.unwrapped.get_action_meanings().index("RIGHT")
    elif keys[pygame.K_UP]:
        action = env.unwrapped.get_action_meanings().index("UP")
    elif keys[pygame.K_SPACE]:
        action = env.unwrapped.get_action_meanings().index("FIRE")
    elif keys[pygame.K_DOWN]:
        action = env.unwrapped.get_action_meanings().index("DOWN")


    return action



def record_human_demo(
    env_id="ALE/Pitfall-v5",
    output="human_demos.npz",
    max_steps=5000,
):
    
    pygame.init()
    screen = pygame.display.set_mode((200, 200))  # ventana mínima
    pygame.display.set_caption("Control Pitfall")

    env = make_env(
        env_id,
        seed=42,
        render_mode="human"
    )
    print(env.unwrapped.get_action_meanings())
    
    KEY_TO_ACTION = build_key_to_action(env)

    obs_buf = []
    act_buf = []

    obs, _ = env.reset()

    total_reward = 0

    for step in range(max_steps):
        action = get_action(env)

        obs_buf.append(obs)
        act_buf.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        print(
            f"\rStep: {step} | "
            f"Reward: {total_reward}",
            end=""
        )

        if terminated or truncated:

            print(
                f"\nEpisode finished "
                f"with reward {total_reward}"
            )

            obs, _ = env.reset()

            total_reward = 0

    env.close()

    demos = {
        "observations": np.array(
            obs_buf,
            dtype=np.float32
        ),
        "actions": np.array(
            act_buf,
            dtype=np.int64
        ),
    }

    np.savez_compressed(
        output,
        **demos
    )

    print(f"\nSaved demos to {output}")


if __name__ == "__main__":

    record_human_demo()