from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


wrapped_env = OT2Env(gym.Env)


check_env(wrapped_env)
