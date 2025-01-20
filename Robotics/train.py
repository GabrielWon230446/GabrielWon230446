import gymnasium as gym
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register
from clearml import Task
import argparse
from typing_extensions import TypeIs
import tensorflow


task = Task.init(project_name='Mentor Group J/Group 3',
                    task_name='iteration 2')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--batch_size", type=int, default=64)
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

def create_env():
    """Create and wrap the OT2 environment."""
    env = gym.make('OT2Env-v0')
    env = Monitor(env)
    return env

class BestDistanceCallback(BaseCallback):
    """
    Custom callback to track and log the best distance achieved during training.
    """
    def __init__(self):
        super(BestDistanceCallback, self).__init__()
        self.best_distance = float("inf")  # Initialize with infinity

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "distance" in info:
                current_distance = info["distance"]
                if isinstance(current_distance, (int, float)):
                    if current_distance < self.best_distance:
                        self.best_distance = current_distance
                        # Log to WandB
                        wandb.log({"best_distance": self.best_distance})
        return True
    
def train():
    """Main training function."""
    # Create environment
    env = create_env()
    eval_env = create_env()

    # Initialize WandB
    run = wandb.init(
        project="iteration 2",
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
    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=10,
        eval_freq=5000,
        log_path="eval_logs",
        best_model_save_path="best_model",
        deterministic=True
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2
    )
    
    best_distance_callback = BestDistanceCallback()

    # Training
    try:
        model.learn(
            total_timesteps=100000,
            callback=[eval_callback, wandb_callback, best_distance_callback],
            progress_bar=True
        )
        
        # Save final model
        model.save("final_model/ppo_ot2")
        
        print("\nTraining completed!")
        print(f"Best distance achieved: {best_distance_callback.best_distance:.6f}m")
        print(f"Best model saved to: best_model/best_model")
        
        # Log best distance to WandB
        wandb.log({"best_distance": best_distance_callback.best_distance})
        
    except Exception as e:
        print(f"Training interrupted: {e}")
        
    finally:
        # Clean up
        env.close()
        eval_env.close()
        wandb.finish()

if __name__ == "__main__":
    train()