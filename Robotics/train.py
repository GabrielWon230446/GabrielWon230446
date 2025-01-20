import gymnasium as gym
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
from clearml import Task
import argparse
from typing_extensions import TypeIs
import tensorflow
import os
from ot2_gym_wrapper import OT2Env

# Activate wandb logging
os.environ['WANDB_API_KEY'] = '1f9a653a148ce2cdf8e255b5baa6fed567eafa83'


task = Task.init(project_name='Mentor Group J/Group 3',
                    task_name='iteration 4')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)

args = parser.parse_args()

# Register the wrapper with Gymnasium
register(
    id='OT2Env-v0',
    entry_point='ot2_gym_wrapper:OT2Env',
    kwargs={'render': False, 'max_steps': 1000}
)

def train():
    """Main training function."""
    # Create environment
    """Create and wrap the OT2 environment."""
    env = OT2Env(max_steps=1000)

    # Initialize WandB
    run = wandb.init(
        project="iteration 4",
        sync_tensorboard=True
    )
    # Initialize model
    model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}",)
    
    
    # Create callbacks
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    wandb_callback = WandbCallback(model_save_freq=10000,
                                    model_save_path=f"models/{run.id}",
                                    verbose=2,
                                    )

    # Training
    try:
        model.learn(
            total_timesteps=4000000,
            callback=[wandb_callback],
            progress_bar=True, 
            reset_num_timesteps=False,
            tb_log_name=f"runs/{run.id}"
        )
        
        # Save final model
        model.save("final_model/ppo_ot2")
        
        print("\nTraining completed!")
        print(f"Best model saved to: best_model/best_model")
        
    except Exception as e:
        print(f"Training interrupted: {e}")
        
    finally:
        # Clean up
        env.close()
        wandb.finish()

if __name__ == "__main__":
    train()