import numpy as np
from pid_controller import PIDController
from ot2_gym_wrapper import OT2Env

if __name__ == "__main__":
    # Initialize the environment
    env = OT2Env(max_steps=1000)

    # Initialize the PID controller
    pid = PIDController(kp=0.1, ki=0.1, kd=0.02)

    # Reset the environment and get the initial observation
    observation, _ = env.reset()
    current_position = observation[:3]
    goal_position = observation[3:]

    done = False
    dt = 1.6  # Time step in seconds
    total_reward = 0

    while not done:
        # Compute the action using the PID controller
        action = pid.compute_action(current_position, goal_position, dt)

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Update the current position
        current_position = observation[:3]

        # Calculate accuracy (distance to the goal)
        accuracy = np.linalg.norm(current_position - goal_position)

        # Check if the episode is finished
        done = terminated or truncated

        print(f"Action: {action}, Current Position: {current_position}, Reward: {reward}, Accuracy: {accuracy:.6f} m")

    print(f"Final Accuracy: {accuracy:.6f} m")
    print(f"PID Gains: kp={pid.kp}, ki={pid.ki}, kd={pid.kd}")
    env.close()
