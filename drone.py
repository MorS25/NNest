import numpy as np

class Drone(object):
    time_step = 0.01

    def __init__(self, time, reference=None, x0=None):
        if x0 is None:
            x0 = np.random.rand(1,2)
        if reference is None:
            reference = np.zeros((2, time.size))

        self.time = time
        self.k = 0
        self.reference = reference
        self.xx = np.zeros((2, time.size))
        self.xx[:,0] = x0

    def step(self, u):
        self.k += 1
        dt = self.time_step
        A = np.matrix([[1, dt],
                       [0, 1]])
        B = np.matrix([[0],
                       [dt]])
        self.xx[:,self.k,None] = A*self.xx[:,self.k-1,None]+B*u

    def getCurrentState(self):
        return self.xx[:, self.k]

    def get_scaled_state(self):
        '''Get full state, scaled into (approximately) [0, 1].'''
        states = self.xx[:, self.k].ravel().tolist()
        refs = self.reference[:, self.k].ravel().tolist()
        states.extend(refs)
        return states

    def diverges(self):
        return abs(self.xx[0,self.k]) > 5

    def currentSquareError(self):
        return np.linalg.norm(self.xx[:,self.k-1] - self.reference[:, self.k-1])

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

def generateSineReference(time, T=3):
    return 0.1*np.vstack((np.sin(2*np.pi/T*time), 2*np.pi/T*np.cos(2*np.pi/T*time)))
