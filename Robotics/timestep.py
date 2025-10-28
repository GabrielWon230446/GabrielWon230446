import time
from sim_class import Simulation

sim = Simulation(num_agents=1)
start_time = time.time()
status = sim.run([[0, 0, 0, 0]])
actual_dt = time.time() - start_time
print(f"One sim step actually takes: {actual_dt:.3f} seconds")