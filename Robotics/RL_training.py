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

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='Mentor Group J/Group 3', # NB: Replace YourName with your own name
                    task_name='PPO_OT2_2')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=15)
parser.add_argument("--gamma", type=float, default=0.96)

args = parser.parse_args()

# Register the wrapper with Gymnasium
register(
    id='OT2Env-v0',
    entry_point='ot2_gym_wrapper:OT2Env',
    kwargs={'render': False, 'max_steps': 1000}
)

# Initialize Weights & Biases
run = wandb.init(project="OT2-rl", sync_tensorboard=True)

# Creat the environment
env = OT2Env(gym.Env)

# Define a PPO model
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            gamma=args.gamma,
            tensorboard_log=f"runs/{run.id}",)

# Evaluate the policy every 10,000 steps
eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=1000)

# Integrate W&B
wandb_callback = WandbCallback()

# Combine callbacks
callbacks = [eval_callback, wandb_callback]

# variable for how often to save the model
time_steps = 1000000
for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{time_steps*(i+1)}")

#Test the trained model
obs = env.reset()[0] # the reset() method only returns the observation (obs = vec_env.reset()) and not a tuple, the info at reset are stored in vec_env.reset_infos
for i in range(1000):
    action, _ = model.predict(obs,deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()
    time.sleep(0.025)
    if done:
        env.reset()