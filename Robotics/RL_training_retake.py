import gymnasium as gym
from sim_class import Simulation
from ot2_gym_wrapper import OT2Env
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
import time
from clearml import Task
import tensorflow

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=4096)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.96)

args = parser.parse_args()

# Register the wrapper with Gymnasium
register(
    id='OT2Env-v0',
    entry_point='ot2_gym_wrapper:OT2Env',
    kwargs={'render': False, 'max_steps': 4096}
)

# Initialize Weights & Biases
run = wandb.init(project="OT2-rl-retake", sync_tensorboard=True)

# Creat the environment
env = OT2Env(render=False, max_steps=4096)

# Define a PPO model
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            gamma=args.gamma,
            tensorboard_log=f"runs/{run.id}",)

# Evaluate the policy each time the model is updated
eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=4096)

# Integrate W&B
wandb_callback = WandbCallback()

# Combine callbacks
callbacks = [eval_callback, wandb_callback]

# Set total timesteps for entire training run
total_timesteps = 8192000

# Train the model in one go
model.learn(
    total_timesteps=total_timesteps,
    callback=callbacks,  # use both EvalCallback and WandbCallback
    progress_bar=True,
    tb_log_name=f"runs/{run.id}",
    reset_num_timesteps=False
)

# Save final model after training
model.save(f"models/{run.id}/final_model")