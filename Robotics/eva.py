# Import necessary libraries
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env
import numpy as np

# Load the trained model
model = PPO.load("ppo_ot2_model")

# Define the evaluation
accuracy_within_1mm = 0
total_episodes = 100

# Initialize the environment
env = OT2Env(render=False)

for episode in range(total_episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        # Replace random actions with the trained model's predictions
        action, _ = model.predict(obs, deterministic=True)  
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Check if the pipette position is within 1 mm accuracy
        if info.get("within_accuracy"):  # Ensure `within_accuracy` is logged in `info`
            accuracy_within_1mm += 1

# Calculate and print results
accuracy_percentage = (accuracy_within_1mm / total_episodes) * 100
print(f"Accuracy within 1 mm: {accuracy_percentage:.2f}%")
