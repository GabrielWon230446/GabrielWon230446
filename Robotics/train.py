import gymnasium as gym
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register
from clearml import Task
import argparse

'''task = Task.init(project_name='Mentor Group J/Group 3',
                    task_name='iteration 2')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")'''

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.96)

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

class BestDistanceCallback(EvalCallback):
    """Custom callback to track best distance achieved."""
    def __init__(self, eval_env, n_eval_episodes=100, eval_freq=5000, 
                 log_path=None, best_model_save_path=None, verbose=1):
        super().__init__(eval_env, n_eval_episodes=n_eval_episodes, 
                        eval_freq=eval_freq, log_path=log_path,
                        best_model_save_path=best_model_save_path, verbose=verbose)
        self.best_distance = float('inf')
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            distances = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    if done:
                        distances.append(info['distance'])
            
            mean_distance = np.mean(distances)
            min_distance = np.min(distances)
            
            if min_distance < self.best_distance:
                self.best_distance = min_distance
                if self.best_model_save_path is not None:
                    self.model.save(f"{self.best_model_save_path}/best_model")
            
            wandb.log({
                'eval/mean_distance': mean_distance,
                'eval/best_distance_so_far': self.best_distance,
                'eval/min_distance_this_eval': min_distance
            })
            
            if self.verbose > 0:
                print(f"Eval num {len(self.evaluations_timesteps)}")
                print(f"Current mean distance: {mean_distance:.6f}m")
                print(f"Best distance so far: {self.best_distance:.6f}m")
        
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
    
    # Training
    try:
        model.learn(
            total_timesteps=100000,
            callback=[eval_callback, wandb_callback],
            progress_bar=True
        )
        
        # Save final model
        model.save("final_model/ppo_ot2")
        
    except Exception as e:
        print(f"Training interrupted: {e}")
        
    finally:
        # Clean up
        env.close()
        eval_env.close()
        wandb.finish()

if __name__ == "__main__":
    train()