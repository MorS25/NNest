from __future__ import print_function

import pickle
import drone
import numpy as np
from neat import nn
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot, scatter, grid

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

figure(1)
ioff()
clf()
hold(True)
grid()
plot(sim.time, sim.reference[0,:], 'r')
plot(sim.time, sim.xx[0,:], 'b')
ion()
draw()
ioff()
show()
