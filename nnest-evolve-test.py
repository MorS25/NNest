from __future__ import print_function

import pickle
import drone
import numpy as np
from neat import nn
import matplotlib.pyplot as plt

# load the winner
with open('nn_winner_genome.gen', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

net = nn.create_feed_forward_phenotype(c)

time = np.arange(0,10,drone.Drone.time_step)
reference = drone.generateSineReference(time)
sim = drone.Drone(time, reference)

for s in range(time.size-1):
    inputs = sim.get_scaled_state()
    action = net.serial_activate(inputs)
    force = drone.continuous_actuator_force(action)
    sim.step(force)


plt.figure(1)
plt.subplot(311)
plt.hold(True)
plt.grid()
plt.plot(sim.time, sim.reference[0,:], 'r')
plt.plot(sim.time, sim.xx[0,:], 'b')
plt.subplot(312)
plt.hold(True)
plt.grid()
plt.plot(sim.time, sim.reference[1,:], 'r')
plt.plot(sim.time, sim.xx[1,:], 'b')
plt.subplot(313)
plt.hold(True)
plt.grid()
plt.plot(sim.time, sim.uu, 'c')
plt.plot(sim.time, sim.perturbation, 'm')
plt.show()
