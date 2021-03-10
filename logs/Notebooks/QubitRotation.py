################################################################################
##                      BASIC TUTORIAL: QUBIT ROTATIONS                       ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 10/03/21
## DESCRIPTION: In this program, I recreate the basic tutorial from Xandau docs.
##              I like that it introduces optimization from the start

################################################################################
##                        IMPORTING PENNYLANE AND NUMPY                       ##
################################################################################
import pennylane as qml
from pennylane import numpy as np               ## ALWAYS IMPORT WRAPPERS
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

################################################################################
##                        INITIALIZING A QUANTUM DEVICE                       ##
################################################################################
device = qml.device('default.qubit',wires=1)

################################################################################
##                              DEFINING A QNODE                              ##
################################################################################
@qml.qnode(device)
def circuit(phis):
    ## Qubit indexing starts from 0. Multiqubit gate takes a list on kwarg wires
    qml.RX(phis[0], wires=0)                    ## Perform X rotation
    qml.RY(phis[1], wires=0)                    ## Perform Y rotation
    return qml.expval(qml.PauliZ(0))            ## Return expectation value
## Testing the function
print(circuit([0.54,0.12]))

################################################################################
##                                WRAPP QNODE                                 ##
################################################################################
def cost_function(vars):
    return circuit(vars)
## Testing the wrapper
print(cost_function([0.54,0.12]))

################################################################################
##                          CREATE OPTIMIZER OBJECT                           ##
################################################################################
optimizer = qml.GradientDescentOptimizer(stepsize = 0.4)

################################################################################
##                           ITERATE UNTIL MINIMUM                            ##
################################################################################
NUMSTEPS = 100                                  ## Number of iterations
params = np.array([0.011, 0.012])               ## Initial guess for params
## I will store and plot optimization results
step = [] * NUMSTEPS
phi1 = [] * NUMSTEPS
phi2 = [] * NUMSTEPS
cost = [] * NUMSTEPS
## Start iteration
for i in range(NUMSTEPS):
    ## Update the circuit parameters
    params = optimizer.step(cost_function, params)
    ## Store data for plotting
    phi1.append(params[0])
    phi2.append(params[1])
    cost.append(cost_function(params))
    step.append(i+1)

################################################################################
##                         PLOT RESULTS OF SIMULATION                         ##
################################################################################
plt.plot(step,cost,label=r'$\phi_1$')
plt.legend()
plt.show()
