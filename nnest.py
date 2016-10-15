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


# Time
dt = 0.005
t = arange(0,5,dt)
# Setpoint
x_ref = np.zeros(t.size)
# Perturbation
T = 3
tau = 5*np.sin(2*math.pi/T*t)
# PID Response
K = np.matrix([120, 20])
x = np.zeros((2, t.size))
u = np.zeros(t.size)
x[:,0] = [1,0]
A = np.matrix([[1, dt],
              [0, 1]])
B = np.matrix([[0],
               [dt]])
for k in range(t.size-1):
    u[k+1] = -K*x[:,k,None]
    x[:,k+1,None] = A*x[:,k,None]+B*(u[k+1]+tau[k+1])

# Dataset
ds = SupervisedDataSet(inp=2, target=1)
for i in range(t.size-1):
    ds.addSample((x[0,i],x[1,i]), (u[i+1],))

# Network training
fnn = buildNetwork( ds.indim, 100, 1, bias=True, hiddenclass=SigmoidLayer, outclass=LinearLayer)
trainer = BackpropTrainer( fnn, dataset=ds, momentum=0.1, verbose=True)
trainer.trainUntilConvergence(maxEpochs = 20)

# Network Response
xNN = np.zeros((2, t.size))
uNN = np.zeros(t.size)
xNN[:,0] = [-2,0]
for k in range(t.size-1):
    uNN[k+1] = fnn.activate((xNN[0,k], xNN[1,k]))
    xNN[:,k+1,None] = A*xNN[:,k,None]+B*(uNN[k+1]+tau[k+1])


figure(1)
ioff()
clf()
hold(True)
grid()
plot(t, x_ref, 'r')
plot(t, tau/100, 'c')
plot(t, x[0,:], 'b')
plot(t, xNN[0,:], 'g')
ion()
draw()
ioff()
show()