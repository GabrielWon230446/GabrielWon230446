from sim_class import Simulation
import random

# Store the pipette coordinates in a dictionary for clarity
output = {
    'top_left_front_corner':[], 
    'top_left_back_corner':[],
    'top_right_front_corner':[],
    'top_right_back_corner':[],
    'bottom_left_front_corner':[], 
    'bottom_left_back_corner':[],
    'bottom_right_back_corner':[],
    'bottom_right_front_corner':[], 

}

# Initialize the simulation with a specified number of agents
sim = Simulation(num_agents=1)  # For one robot

# Run the simulation for a specified number of steps
for i in range(200):
    # Move the pipette to the top left back corner
    velocity_x = -1
    velocity_y = -1
    velocity_z = 1
    drop_command = 0


    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    state = sim.run(actions)
output['top_left_back_corner'] = state['robotId_1']['pipette_position']

for i in range(100):
    # Move the pipette to the bottom left back corner
    velocity_x = -1
    velocity_y = -1
    velocity_z = -1
    drop_command = 0


    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    state = sim.run(actions)
output['bottom_left_back_corner'] = state['robotId_1']['pipette_position']

for i in range(200):
    # Move the pipette to the top left front corner
    velocity_x = 1
    velocity_y = -1
    velocity_z = 1
    drop_command = 0


    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    state = sim.run(actions)
output['top_left_front_corner'] = state['robotId_1']['pipette_position']

for i in range(100):
    # Move the pipette to the bottom left front corner
    velocity_x = 1
    velocity_y = -1
    velocity_z = -1
    drop_command = 0


    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    state = sim.run(actions)
output['bottom_left_front_corner'] = state['robotId_1']['pipette_position']

for i in range(200):
    # Move the pipette to the bottom right front corner
    velocity_x = 1
    velocity_y = 1
    velocity_z = -1
    drop_command = 0


    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    state = sim.run(actions)
output['bottom_right_front_corner'] = state['robotId_1']['pipette_position']

for i in range(100):
    # Move the pipette to the top right front corner
    velocity_x = 0
    velocity_y = 0
    velocity_z = 1
    drop_command = 0


    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    state = sim.run(actions)
output['top_right_front_corner'] = state['robotId_1']['pipette_position']

for i in range(200):
    # Move the pipette to the top right back corner
    velocity_x = -1
    velocity_y = 0
    velocity_z = 0
    drop_command = 0


    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    state = sim.run(actions)
output['top_right_back_corner'] = state['robotId_1']['pipette_position']

for i in range(100):
    # Move the pipette to the bottom right back corner
    velocity_x = 0
    velocity_y = 0
    velocity_z = -1
    drop_command = 0


    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    state = sim.run(actions)
output['bottom_right_back_corner'] = state['robotId_1']['pipette_position']

print(output)

