#! /usr/bin/env python

import math
import numpy as np
import drone
import matplotlib.pyplot as plt

DEG_TO_RAD = math.pi/180

class DroneMDP(object):
    def __init__(self, obstacle=100):
        self.drone = drone.Drone()
        self.obstacle = obstacle

    def step(self, action, params):
        self.mdpAction(action)
        reward = self.mdpReward(params)
        state = self.mdpState(params)
        return reward, state

    def reset(self):
        self.drone.reset()

    """ x, d, ucmd"""
    def mdpState(self, params):
        state = [params['ucmd'], self.obstacle - self.drone.x[0,0], self.drone.x[1,0]]
        return np.array(state).reshape((1,3))

    def mdpAction(self, action):
        u = (action-1)*10*DEG_TO_RAD
        self.drone.step(u)

    def mdpReward(self, params):
        alpha = 1.
        beta = 5.
        dObs = self.obstacle-self.drone.x[0,0]
        reward = -500.
        if dObs > np.sqrt(beta/500.):
            reward = -alpha*(params["ucmd"] - self.drone.u)**2 -beta/(dObs**2)
        return reward


def test():
    dMDP = DroneMDP()

    time = np.arange(0,10,drone.Drone.time_step)
    params = {"ucmd":10*DEG_TO_RAD}

    rr = np.zeros(np.size(time))
    ss = np.zeros((np.size(time), np.size(dMDP.mdpState(params))))
    for t in range(time.size):
        action = 2 #*math.cos(2*math.pi*time[t]/8)
        reward, state = dMDP.step(action, params)
        rr[t] = reward
        ss[t] = state

    plt.figure(1)
    plt.subplot(211)
    plt.grid()
    plt.plot(time, ss)
    plt.subplot(212)
    plt.grid()
    plt.plot(time, rr, 'b')
    plt.show()

if __name__ == "__main__":
    test()