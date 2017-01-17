from __future__ import print_function

import os
import pickle

import drone
import numpy as np

from neat import nn, parallel, population, visualize
from neat.config import Config
from neat.math_util import mean

runs_per_net = 3

def evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)

    fitnesses = []

    for runs in range(runs_per_net):
        time = np.arange(0,5,drone.Drone.time_step)
        reference = drone.generateSineReference(time)
        sim = drone.Drone(time, reference)

        fitness = 0.0
        for s in range(time.size-1):
            inputs = sim.get_scaled_state()
            action = net.serial_activate(inputs)

            # Apply action
            force = drone.continuous_actuator_force(action)
            sim.step(force)

            if sim.diverges():
                fitness += -10*sim.currentSquareError()*(time.size-runs)
                break

            fitness += -sim.currentSquareError()

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'nn_config'))

    pop = population.Population(config)
    pe = parallel.ParallelEvaluator(4, evaluate_genome)
    pop.run(pe.evaluate, 200)

    # Save the winner.
    print('Number of evaluations: {0:d}'.format(pop.total_evaluations))
    winner = pop.statistics.best_genome()
    with open('nn_winner_genome.gen', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    # Plot the evolution of the best/average fitness.
    visualize.plot_stats(pop.statistics, ylog=True, filename="nn_fitness.svg")
    # Visualizes speciation
    visualize.plot_species(pop.statistics, filename="nn_speciation.svg")
    # Visualize the best network.
    visualize.draw_net(winner, view=True, filename="nn_winner.gv")
    visualize.draw_net(winner, view=True, filename="nn_winner-enabled.gv", show_disabled=False)
    visualize.draw_net(winner, view=True, filename="nn_winner-enabled-pruned.gv", show_disabled=False, prune_unused=True)

if __name__ == '__main__':
    run()