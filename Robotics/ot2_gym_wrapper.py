import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
from typing_extensions import TypeIs

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=True)

        # Define action and observation space
        # They must be gym.spaces objects
        
        # YOUR CODE HERE
        low = np.array([-1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1], dtype=np.float32)
        shape = (3,)
        dtype = np.float32
        self.action_space = spaces.Box(low, high, shape, dtype)
        
        # YOUR CODE HERE
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

        # keep track of the number of steps
        self.steps = 0

    def pipette_position_extractor(self, raw_observation, goal_position):
        # Extract the pipette position from the current observation and append the goal position
        robotId = list(raw_observation.keys())[0]
        pipette_position = np.array(raw_observation[robotId]['pipette_position'])
        observation = np.append(pipette_position, goal_position)
        # Ensure the array is of type np.float32
        observation = observation.astype(np.float32)
        return observation

    def reset(self, seed=None, options=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area
        # YOUR CODE HERE
        # Define the working envelope corners
        envelope = {
            'top_left_front_corner': [0.2534, -0.1705, 0.2895],
            'top_left_back_corner': [-0.187, -0.1705, 0.2895],
            'top_right_front_corner': [0.253, 0.2195, 0.2895],
            'top_right_back_corner': [-0.1871, 0.2195, 0.2895],
            'bottom_left_front_corner': [0.253, -0.1705, 0.1685],
            'bottom_left_back_corner': [-0.187, -0.1709, 0.1685],
            'bottom_right_back_corner': [-0.1869, 0.2195, 0.1687],
            'bottom_right_front_corner': [0.253, 0.2195, 0.169],
        }

        # Extract the range of corner coordinates
        coordinates = np.array(list(envelope.values()))
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

        # Generate random goal positions within the extracted range
        goal_x = np.random.uniform(x_min, x_max)
        goal_y = np.random.uniform(y_min, y_max)
        goal_z = np.random.uniform(z_min, z_max)

        self.goal_position = [goal_x, goal_y, goal_z]
        
        # Call the environment reset function
        raw_observation = self.sim.reset(num_agents=1)
        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        
        # YOUR CODE HERE
        # Extract the pipette position from the reset observation
        observation = self.pipette_position_extractor(raw_observation, self.goal_position)

        # Reset the number of steps
        self.steps = 0

        return observation, {}

    def step(self, action):
        # Execute one time step within the environment
        action = np.append(action, 0)  # Append 0 for the drop action

        # Call the environment step function
        raw_observation = self.sim.run([action])
        
        # Process the observation
        observation = self.pipette_position_extractor(raw_observation, self.goal_position)

        # Now extract pipette and goal positions
        pipette_position = observation[:3]  # First 3 elements should be pipette position
        goal_position = observation[3:]    # Last 3 elements should be goal position
        
        # Calculate the Euclidean distance between the pipette position and goal position 
        distance = np.linalg.norm(pipette_position - goal_position)
        reward = -distance**2
        
        # Ensure the reward is a float
        reward = float(reward)
        
        # Check if the accuracy is within 1mm
        threshold = 0.001 # 10mm accuracy
        within_accuracy = distance <= threshold
        if within_accuracy:
            terminated = True
            reward += 10  # Add positive reward for completing the task
        else:
            terminated = False

        # Check if the episode should be truncated
        truncated = self.steps >=self.max_steps

        # Increment the number of steps
        self.steps += 1

        info = {"distance": distance, "within_accuracy": terminated, "reached max steps": truncated}

        return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()