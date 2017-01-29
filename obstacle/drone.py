import numpy as np

class Drone(object):
    time_step = 0.01

    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.zeros((4,1))

        self.x = x0

    def step(self, u):
        dt = self.time_step
        w0 = 11
        zeta = 0.9
        Cx = 0.3
        G = 9.81
        A = np.matrix([[1, dt, 0, 0],
                       [0, 1-Cx*dt, G*dt, 0],
                       [0, 0, 1, dt],
                       [0, 0, -dt*w0**2, 1-2*zeta*w0*dt]])
        B = np.matrix([[0],
                       [0],
                       [0],
                       [dt*w0**2]])
        self.u = u
        self.x= A*self.x+B*u

