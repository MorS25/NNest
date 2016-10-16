import numpy as np

class Drone(object):
    time_step = 0.01

    def __init__(self, x=None, steps=0, x_limit=[5,5]):
        self.x_limit = x_limit

        if x is None:
            x = np.random.rand(2,1)

        self.k = 0
        self.t = 0
        self.x = x
        self.xx = np.zeros((2,steps))
        self.tt = np.zeros(steps)

    def step(self, u):
        dt = self.time_step
        A = np.matrix([[1, dt],
                       [0, 1]])
        B = np.matrix([[0],
                       [dt]])
        self.x[:,0,None] = A*self.x[:,0,None]+B*u
        self.xx[:,self.k,None] = self.x
        self.t += dt
        self.tt[self.k] = self.t
        self.k += 1

    def get_scaled_state(self):
        '''Get full state, scaled into (approximately) [0, 1].'''
        return self.x.ravel().tolist()

    def get_state(self):
        return self.x.tolist()


def continuous_actuator_force(action):
    return 50*(action[0]-0.5)


def noisy_continuous_actuator_force(action):
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0


def discrete_actuator_force(action):
    return 10.0 if action[0] > 0.5 else -10.0


def noisy_discrete_actuator_force(action):
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0
