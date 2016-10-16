from __future__ import print_function

import pickle
import drone
from neat import nn
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot, scatter, grid

# load the winner
with open('nn_winner_genome', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

net = nn.create_feed_forward_phenotype(c)

num_balanced = 0
num_steps = 10000
sim = drone.Drone(steps=num_steps)

print()
print("Initial conditions:")
print("        x = {0:.4f}".format(sim.x.item(0)))
print("       dx = {0:.4f}".format(sim.x.item(1)))
print()


for s in range(num_steps):
    inputs = sim.get_scaled_state()
    action = net.serial_activate(inputs)

    force = drone.continuous_actuator_force(action)
    sim.step(force)

    if abs(sim.x[0]) >= sim.x_limit[0]:
        break

    num_balanced += 1

print()
print("Final conditions:")
print("        x = {0:.4f}".format(sim.x.item(0)))
print("       dx = {0:.4f}".format(sim.x.item(1)))
print()

figure(1)
ioff()
clf()
hold(True)
grid()
plot(sim.tt, sim.xx[0,:], 'b')
ion()
draw()
ioff()
show()
