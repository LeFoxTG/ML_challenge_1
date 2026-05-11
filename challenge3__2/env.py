import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
gym.register_envs(ale_py)

def make_env(env_id: str, seed: int = 0,  render_mode=None):
    env = gym.make(
        env_id,
        frameskip=1,
        render_mode=render_mode
    )

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True,
    )

    env = FrameStackObservation(env, 4)
    env.reset(seed=seed)

    return env