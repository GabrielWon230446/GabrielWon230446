from pathlib import Path
import numpy as np
from ot2_gym_wrapper import OT2Env
from stable_baselines3 import PPO

# Resolve the test script's directory
test_script_dir = Path(__file__).parent

# Construct the absolute path to the model file
model_path = (
    test_script_dir.parent.parent.parent / "models" / "y4qlp952" / "100000_model.zip"
)

# Debug: Print the resolved path
print(f"Resolved model path: {model_path}")

# Check if the model file exists
if not model_path.is_file():
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load the model
model = PPO.load(model_path)
print("Model loaded successfully!")
print("Model's observation space:", model.observation_space)

# Define test parameters
num_tests = 1000  # Number of random test episodes
threshold = 0.001  # 1mm accuracy threshold

# Environment for testing
env = OT2Env()
success_count = 0

for _ in range(num_tests):
    # Reset the environment with a new random goal
    obs, _ = env.reset() # returns Obervation, {}

    terminated = False
    truncated = False
    steps = 0
    max_steps = env.max_steps  # Use the max_steps defined in the environment

    while not (terminated or truncated):
        # Predict action using the trained model
        action, _ = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1

        # Check if the goal was reached
        if terminated:
            success_count += 1
            break

# Calculate the accuracy as a percentage
accuracy = (success_count / num_tests) * 100
print(f"Model accuracy over {num_tests} tests: {accuracy:.2f}%")
