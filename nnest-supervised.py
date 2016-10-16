from pybrain.datasets            import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer, SigmoidLayer, LinearLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot, scatter, grid
from mpl_toolkits.mplot3d import Axes3D
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import numpy as np
import math

def gaussMarkovProcess(t, tau, sigma):
    noise = np.random.normal(0, sigma, t.size)
    gamma = np.zeros(t.size)
    for k in range(t.size-1):
        gamma[k+1]=gamma[k]+dt*(-1/tau*gamma[k]+noise[k])
    return gamma

def PIDcontroller(K, x):
    return -K*x

def runSimulation(t, controller, x0=None, perturbation=None):
    if perturbation is None:
        perturbation = np.zeros(t.size)
    A = np.matrix([[1, dt],
                  [0, 1]])
    B = np.matrix([[0],
                   [dt]])
    x = np.zeros((2, t.size))
    if x0 is not None:
        x[:,0] = x0
    u = np.zeros(t.size)
    for k in range(t.size-1):
        u[k+1] = controller(x[:,k,None])
        x[:,k+1,None] = A*x[:,k,None]+B*(u[k+1]+perturbation[k+1])
    return (x, u)


# Time
dt = 0.005
t = arange(0,5,dt)

# Setpoint
x_ref = np.zeros(t.size)

# Perturbation
wind = gaussMarkovProcess(t, tau=3, sigma=500)

# PID Response
K = np.matrix([120, 20])
pid = lambda x: PIDcontroller(K, x)
(x, u) = runSimulation(t, pid, x0 = [1,0], perturbation=wind)

# Dataset
ds = SupervisedDataSet(inp=2, target=1)
for i in range(t.size-1):
    ds.addSample((x[0,i],x[1,i]), (u[i+1],))

# Network training
fnn = buildNetwork( ds.indim, 100, 1, bias=True, hiddenclass=SigmoidLayer, outclass=LinearLayer)
trainer = BackpropTrainer( fnn, dataset=ds, momentum=0.1, verbose=True)
trainer.trainUntilConvergence(maxEpochs = 20)

# Network Response
NNcontroller = lambda x: fnn.activate((x[0], x[1]))
(xNN, uNN) = runSimulation(t, NNcontroller, x0=[-2,0], perturbation=wind)

figure(1)
ioff()
clf()
hold(True)
grid()
plot(t, x_ref, 'r')
plot(t, wind/100, 'c')
plot(t, x[0,:], 'b')
plot(t, xNN[0,:], 'g')
ion()
draw()
ioff()
show()