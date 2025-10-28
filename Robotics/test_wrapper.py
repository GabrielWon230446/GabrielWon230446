from ot2_gym_wrapper import OT2Env

# Load the custom environment
env = OT2Env(render=False, max_steps=1000)

# Number of episodes
num_episodes = 5

for episode in range(num_episodes):
    obs, _ = env.reset()  # Gymnasium reset returns (obs, info)
    done = False
    step = 0

    while not done:
        # Take a random action from the environment's action space
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")

        step += 1

        if done:
            print(f"Episode finished after {step} steps. Info: {info}")
            break

env.close()

