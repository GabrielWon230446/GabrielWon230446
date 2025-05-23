# Robotic Simulation with PID and Reinforcement Learning

This project sets up a robotic simulation environment where a robot arm is guided to a goal position using both a classical PID controller and reinforcement learning (RL). The simulation supports sensor feedback for situational awareness and enables training of RL agents for control tasks.

## 📦 Project Structure

| File | Description |
|------|-------------|
| `pid_controller.py` | Implements a PID controller for classical control. |
| `sim_class.py` | Defines the custom simulation environment logic. |
| `eva.py`, `ot2_gym_wrapper.py` | Wrappers and utilities for gym integration. |
| `RL_training.py`, `train.py` | Train RL agents in the environment. |
| `test.py`, `run_pid_with_env.py` | Run evaluation using PID or RL policies. |
| `custom.urdf`, `ot_2_simulation_v6.urdf` | URDF files describing the robot setup. |
| `clearml.png`, `RL_1000000.png` | Visual results and experiment tracking screenshot. |
| `README.md` | This file. |

---

##  Features
- Simulated robotic environment using a custom class.
- PID-based control for reaching 3D target positions and execute actions such as dropping the liquid on given coordinates.
- RL training using stable-baselines3 PPO.
- Real-time sensor feedback via position readings.
- ClearML integration for experiment tracking.

## Calculate the working envolope:
Move the tip of the pipette to each corner of the cube that forms the working envelope by adjusting the motor velocities for each axis and recording the co-ordinates at each of the 8 points.

## Usage:
Run the task9.py file to calculate the working envolope

## Dependencies:
- python (3.10.15)
- pybullet (3.25)

## OT-2 Digital Twin:
The OT-2 Digital Twin is a virtual representation of the Opentrons OT-2, a popular robotic liquid handling system used in labs. To use the OT-2 Digital Twin with PyBullet, you need to first clone the GitHub repository that contains the necessary files:

`git clone https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git`

## Working Envolope Pipette Coordinates
'top_left_front_corner': [0.2534, -0.1705, 0.2895], 'top_left_back_corner': [-0.187, -0.1705, 0.2895], 'top_right_front_corner': [0.253, 0.2195, 0.2895], 'top_right_back_corner': [-0.1871, 0.2195, 0.2895], 'bottom_left_front_corner': [0.253, -0.1705, 0.1685], 'bottom_left_back_corner': [-0.187, -0.1709, 0.1685], 'bottom_right_back_corner': [-0.1869, 0.2195, 0.1687], 'bottom_right_front_corner': [0.253, 0.2195, 0.169]