import numpy as np
import time
import json
from pid_controller import PIDController
from sim_class import Simulation

def gen_goal(): # Generate random goal within the working envelope
    envelope = {
            'top_left_front_corner': [0.2534, -0.1705, 0.2895],
            'top_left_back_corner': [-0.187, -0.1705, 0.2895],
            'top_right_front_corner': [.253, 0.2195, 0.2895],
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

    goal_position = [goal_x, goal_y, goal_z]

    return np.array(goal_position)

if __name__ == "__main__":
    # Initialize the environment
    sim = Simulation(num_agents=1, render=True)

    status = sim.run([[-0.1871, 0.2195, 0.2895, 0]]) # Random position within the working envelope
    robotId = list(status.keys())[0]
    current_position = np.array(status[robotId]['pipette_position'])

    # Initialize the PID controller
    pid_x = PIDController(kp=3, ki=0.1, kd=0.5)
    pid_y = PIDController(kp=3, ki=0.1, kd=0.5)
    pid_z = PIDController(kp=3, ki=0.1, kd=0.5)

    # Reset the environment and generate a new goal position
    goal_position = gen_goal()

    done = False
    dt = 0.005  # Time step in seconds

    # Time and performance tracking
    start_time = time.time()
    step = 0
    accuracy_history = []

    while not done:
        step += 1

        # Compute the action using the PID controller
        action_x = pid_x.compute_action(current_position[0], goal_position[0], dt)
        action_y = pid_y.compute_action(current_position[1], goal_position[1], dt)
        action_z = pid_z.compute_action(current_position[2], goal_position[2], dt)

        action = [action_x, action_y, action_z, 0]

        #action = np.append(action, 0)
        print(action)

        # Run the simulation
        status = sim.run([action])

        # Update the current position
        robotId = list(status.keys())[0]
        current_position = np.array(status[robotId]['pipette_position'])


        # Calculate accuracy (distance to the goal)
        accuracy = np.linalg.norm(current_position - goal_position)

        print(f"Action: {action}, Current Position: {current_position}, Accuracy: {accuracy:.6f} m")

        if accuracy < 0.001:
            done = True # End the simulation if the accuracy requirement is met

    # Final results calculation
    end_time = time.time()
    total_time = end_time - start_time
    final_accuracy = accuracy

    results = {
        "total_steps": step, 
        "final_distance_from_goal": final_accuracy,
        "PID Gains": {
            "kp": pid_x.kp,
            "ki": pid_x.ki, 
            "kd": pid_x.kd # Same gains for each axis
        },
        "total_time_seconds": total_time,
        'goal_position': goal_position.tolist(),
        'final_position': current_position.tolist()
    }
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_filename = f"pid_results_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_filename}")
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total Steps: {results['total_steps']}")
    print(f"Total Time: {results['total_time_seconds']:.2f} seconds") 
    print(f"Final Distance from Goal: {results['final_distance_from_goal']:.6f} m")
    print(f"PID Gains: Kp={results['PID Gains']['kp']}, Ki={results['PID Gains']['ki']}, Kd={results['PID Gains']['kd']}")

    sim.close()