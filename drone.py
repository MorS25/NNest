import numpy as np

class Drone(object):
    time_step = 0.01

    def __init__(self, time, reference=None, x0=None):
        if x0 is None:
            x0 = np.random.rand(1,4)
        if reference is None:
            reference = np.zeros((4, time.size))

        self.time = time
        self.k = 0
        self.reference = reference
        self.xx = np.zeros((4, time.size))
        self.xx[:,0] = x0
        self.uu = np.zeros(time.size)
        self.perturbation = np.zeros(time.size)

    def step(self, u):
        self.k += 1
        dt = self.time_step
        w0 = 90
        zeta = 0.9
        # u = 1*(self.reference[0, self.k-1,None] - self.xx[0,self.k-1,None])+2*(self.reference[1, self.k-1,None] - self.xx[1,self.k-1,None])
        A = np.matrix([[1, dt, 0, 0],
                       [0, 1, dt, 0],
                       [0, 0, 1, dt],
                       [0, 0, -dt*w0**2, 1-2*zeta*w0*dt]])
        B = np.matrix([[0],
                       [0],
                       [0],
                       [dt*w0**2]])
        self.uu[self.k] = u
        self.perturbation[self.k] = self.generatePerturbation()
        self.xx[:,self.k,None] = A*self.xx[:,self.k-1,None]+B*(u+self.perturbation[self.k])

    def generatePerturbation(self):
        return 0 #+0*np.clip(300*self.xx[0,self.k-1]*abs(self.xx[0,self.k-1]), -40, 40)

    def getCurrentState(self):
        return self.xx[:, self.k]

    def get_scaled_state(self):
        '''Get full state, scaled into (approximately) [0, 1].'''
        states = self.xx[0:2, self.k].ravel().tolist()
        refs = self.reference[0:2, self.k].ravel().tolist()
        states.extend(refs)
        return states

    def diverges(self):
        return abs(self.xx[0,self.k]) > 5

    def currentSquareError(self):
        return np.linalg.norm(self.xx[0:2,self.k-1] - self.reference[0:2, self.k-1])

def continuous_actuator_force(action):
    return 200*2*(action[0]-0.5)

def noisy_continuous_actuator_force(action):
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0

def discrete_actuator_force(action):
    return 10.0 if action[0] > 0.5 else -10.0

def noisy_discrete_actuator_force(action):
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0

def generateSineReference(time, T=2):
    ref = 0.1*np.sin(2*np.pi/T*time)
    # ref = np.ones(time.size)
    ref = np.sign(ref)*(abs(ref)**1)
    dref = np.insert(np.diff(ref)/np.diff(time),0,0)
    ref = np.vstack((ref, dref))
    return np.vstack((ref, np.zeros((2,time.size))))
