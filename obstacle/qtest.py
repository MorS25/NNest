import droneMDP
import numpy as np
import random
import csv
from nn import neural_net, LossHistory
import os.path
import timeit
from keras.models import load_model
import matplotlib.pyplot as plt

NUM_INPUT = 3
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.

def test_net(model):
    test_frames = 6000  # Number of frames to play.
    dt = 0.1
    time =  np.arange(test_frames)*dt
    rr = np.zeros(test_frames)
    ss = np.zeros((test_frames,3))
    xx = np.zeros((4,test_frames))
    uu = np.zeros(test_frames)
    simulation_score = 0
    epsilon = 0
    # Create a new sim instance.
    drone = droneMDP.DroneMDP()

    simParams = {"ucmd":10*droneMDP.DEG_TO_RAD}
    state = drone.mdpState(simParams)

    start_time = timeit.default_timer()
    # Run the frames.
    for t in range(test_frames):
        simulation_score += dt/(1+(state[0,1] - 1)**2)

        # Choose an action.
        if random.random() < epsilon:
            action = np.random.randint(3) # random
        else:
            # Get Q values for each action.
            qval = model.predict(state, batch_size=1)
            action = (np.argmax(qval))  # best

        # Take action, observe new state and get our treat.
        reward, state = drone.step(action, simParams)


        if state[0,1] < 0:
            # Time it.
            tot_time = timeit.default_timer() - start_time
            fps = simulation_score / tot_time

            # Reset.
            simulation_score = 0
            start_time = timeit.default_timer()
            drone.reset()

        xx[:,t,None] = drone.drone.x
        uu[t] = drone.drone.u
        rr[t] = reward
        ss[t] = state


    plt.figure(1)
    ax1 = plt.subplot(411)
    plt.grid()
    plt.plot(time, xx[0,:])
    plt.subplot(412, sharex=ax1)
    plt.grid()
    plt.plot(time, xx[1,:])
    plt.subplot(413, sharex=ax1)
    plt.grid()
    plt.plot(time, uu, time, ss[:,0])
    plt.subplot(414, sharex=ax1)
    plt.grid()
    plt.plot(time, rr)
    plt.show()
    # print ss
    # np.savetxt('results.csv', ss, delimiter=',')   # X is an array


if __name__ == "__main__":
    model = load_model('saved-models/latest.h5')
    test_net(model)
