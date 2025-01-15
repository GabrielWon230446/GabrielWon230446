import argparse
import os
import time
import wandb
import gymnasium as gym
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback

# Initialize wandb project
wandb.login()
run = wandb.init(project="sb3_pendulum_demo", sync_tensorboard=True)

# Create the environment
env = gym.make('Pendulum-v1', g=9.81)

# Define the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--timesteps", type=int, default=10000)  # Add timesteps argument for flexibility

args = parser.parse_args()

# Hyperparameters (can be customized through command line or default values)
learning_rate = args.learning_rate
batch_size = args.batch_size
n_steps = args.n_steps
n_epochs = args.n_epochs
timesteps = args.timesteps  # Use timesteps from arguments

# Create PPO model with tensorboard logging
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=learning_rate, 
            batch_size=batch_size, 
            n_steps=n_steps, 
            n_epochs=n_epochs, 
            tensorboard_log=f"runs/{run.id}")

# Create a wandb callback to track the training progress
wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path=f"models/{run.id}",
    verbose=2
)

# Start training with wandb callback and progress bar
model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True)

# Test the trained model
obs = env.reset()[0]
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.025)
    if terminated:
        obs = env.reset()

# Save the model after the initial training
model.save(f"models/{run.id}/final_model")

# Continue training in chunks, saving the model after each chunk
for i in range(10):  # This can be adjusted to train in as many chunks as needed
    model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"runs/{run.id}")
    
    # Save the model with a name that includes the current number of timesteps
    model.save(f"models/{run.id}/{timesteps * (i + 1)}_model")
    print(f"Saved model after {timesteps * (i + 1)} timesteps.")
    print(f"Model saved at: models/{run.id}/{timesteps * (i + 1)}_model")
# Finish the wandb run
wandb.finish()

