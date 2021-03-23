################################################################################
##                         VARIATIONAL CLASSIFIER                             ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 12/03/21
## DESCRIPTION: In this program, I recreate the variational classifier tutorial
##              from Xanadu docs. Here, I fit parity function

################################################################################
##                IMPORTING PENNYLANE, NUMPY AND OPTIMIZER                    ##
################################################################################
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

################################################################################
##                  DEFINITION OF QUANTUM LAYER AND DEVICE                    ##
################################################################################
dev = qml.device('default.qubit', wires=4)          ## Using 4 qubits
## The layer is not the primary qnode
def layer(W):
    qml.Rot(W[0,0], W[0,1], W[0,2], wires=0)
    qml.Rot(W[1,0], W[1,1], W[1,2], wires=1)
    qml.Rot(W[2,0], W[2,1], W[2,2], wires=2)
    qml.Rot(W[3,0], W[3,1], W[3,2], wires=3)
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[3,0])

################################################################################
##                         DEFINITION OF QUANTUM NODE                         ##
################################################################################
@qml.qnode(dev)
def DemoQNN(weights, x):
    ## Carry out initialization step
    qml.BasisState(x, wires=[0,1,2,3])
    ## Implement several layers each with its parameters
    for W in weights:
        layer(W)
    ## Return expected value of PauliZ on 0th qubit
    return qml.expval(qml.PauliZ(0))

################################################################################
##                          DEFINITION OF CLASSIFIER                          ##
################################################################################
def classifier(params, x):
    weights = params[0]
    biases = params[1]
    return DemoQNN(weights, x) + biases

################################################################################
##                         DEFINITION OF COST FUNCTION                        ##
################################################################################
def cost(params, X, Labels):
    predicts = [classifier(params,x) for x in X]
    ## Compute square distance
    loss = 0
    for lab, pred in zip(Labels, predicts):
        loss = loss + (lab - pred)**2
    loss = loss / len(Labels)
    ## Return loss
    return loss

################################################################################
##                           DEFINITION OF ACCURACY                           ##
################################################################################
def accuracy(Labels, predicts):
    acc = 0
    for lab, pred in zip(Labels, predicts):
        if abs(lab - pred) < 1e-5:
            acc = acc + 1
    acc = acc / len(Labels)
    return acc

################################################################################
##                             LOAD FITTING DATA                              ##
################################################################################
fit_data = np.loadtxt('varClassData.txt')
## Load strings
X = np.array(fit_data[:,:-1],requires_grad=False)
Y = np.array(fit_data[:,-1], requires_grad=False)
## Shift label data
Y = Y * 2 - np.ones(len(Y))

################################################################################
##               INITIALIZE PARAMETERS FOR ITERATIVE OPTIMIZATION             ##
################################################################################
np.random.seed(0)
num_quibits = 4
num_layers = 2
## Second variable is the bias
init_vals = (0.01 * np.random.randn(num_layers,num_quibits,3), 0.0)

################################################################################
##               INITIALIZE PARAMETERS FOR ITERATIVE OPTIMIZATION             ##
################################################################################
Costs = []
Accs = []
## define the variable that is to be optimized
var = init_vals
## Carry out 25 Gradient descent iterations
batch_size = 5
## Rename optimizer for simplicity
## Not pretty sure how to choose seed
optimizer = NesterovMomentumOptimizer(0.5)
for _ in range(25):
    ## Select a random batch for realism
    rd_index = np.random.randint(0,len(X),(batch_size,))
    ## Select sample input chains
    X_batch = X[rd_index]
    ## Select corresponding labels
    Y_batch = Y[rd_index]
    ## Wrap cost function for optimization
    ## This simulates us not having access to the
    ## entire space of possible inputs
    wrap_cost = lambda params: cost(params,X_batch,Y_batch)
    ## Optimize with Nesterov
    var = optimizer.step(wrap_cost, var)
    ## Store cost
    Costs.append(float(cost(var,X,Y)))
    ## Store accuracy
    predicts = [np.sign(classifier(var,chain)) for chain in X]
    Accs.append(accuracy(Y,predicts))

################################################################################
##               PLOT EVOLUTION OF COST AND ACCURACY WITH ITERATIONS          ##
################################################################################
its = [i+1 for i in range(25)]
fig, ax = plt.subplots(1,2)
## Plot cost function
ax[0].plot(its,Costs)
ax[0].set_title('Cost function')
## Plot accuracy
ax[1].plot(its,Accs)
ax[1].set_title('Accuracy')
plt.show()
