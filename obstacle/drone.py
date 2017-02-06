import numpy as np

class Drone(object):
    time_step = 0.1

    def __init__(self, x0=None):
        if x0 is None:
            x0 = np.zeros((4,1))

        self.x = x0
        self.A = np.matrix([[1.000000000000000,   0.098514888171639,   0.045213264048830,   0.001003157069378],
                            [                0,   0.970445533548508,   0.845049047568987,   0.025350754075136],
                            [                0,                   0,   0.683602870060148,   0.035750187880105],
                            [                0,                   0,  -4.325772733492666,  -0.024250849965924]])
        self.B = np.matrix([[0.003349892738561],
                            [0.121382005394795],
                            [0.316397129939852],
                            [4.325772733492666]])


    def step(self, u):
        # dt = self.time_step
        # w0 = 11
        # zeta = 0.9
        # Cx = 0.3
        # G = 9.81
        # A = np.matrix([[1, dt, 0, 0],
        #                [0, 1-Cx*dt, G*dt, 0],
        #                [0, 0, 1, dt],
        #                [0, 0, -dt*w0**2, 1-2*zeta*w0*dt]])
        # B = np.matrix([[0],
        #                [0],
        #                [0],
        #                [dt*w0**2]])
        self.u = u
        self.x = self.A*self.x+self.B*u

    def reset(self):
        self.x = np.zeros((4,1))

