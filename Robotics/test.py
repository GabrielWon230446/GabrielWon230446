import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import time
from gymnasium.envs.registration import register

# Register the wrapper with Gymnasium
register(
    id='OT2Env-v0',
    entry_point='ot2_gym_wrapper:OT2Env',
    kwargs={'render': False, 'max_steps': 1000}
)

def test_training():
    """Test script to verify the environment and training setup."""
    # Create environment
    env = gym.make('OT2Env-v0')
    
    # Initialize model with same hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1
    )
    
    # Test variables
    max_training_time = 300  # 5 minutes in seconds
    test_timesteps = 10000   # Short training run
    start_time = time.time()
    
    try:
        # Short training run
        model.learn(total_timesteps=test_timesteps)
        
        # Test if training completes within time limit
        training_time = time.time() - start_time
        if training_time > max_training_time:
            raise TimeoutError(f"Training took too long: {training_time:.2f} seconds")
        
        # Test model prediction
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        # Verify action space
        assert len(action) == 3, f"Unexpected action shape: {action.shape}"
        assert all(-1 <= a <= 1 for a in action), f"Action out of bounds: {action}"
        
        # Test environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify observation space
        assert len(obs) == 6, f"Unexpected observation shape: {obs.shape}"
        
        # Verify reward is numerical
        assert isinstance(reward, (int, float)), f"Invalid reward type: {type(reward)}"
        
        print("All tests passed successfully!")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Final distance: {info['distance']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        
    finally:
        env.close()

if __name__ == "__main__":
    test_training()