################################################################################
##                 BASIC QUANTUM CHEMISTRY TUTORIAL PENNYLANE                 ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 23/04/21
## DESCRIPTION: This is my first approximation to quantum chemistry on a
##              quantum programming platform. I have already read about the
##              subject, but this is my first experimentation.

################################################################################
##                            NECESSARY IMPORTS                               ##
################################################################################
import pennylane as qml
from pennylane import numpy as np

################################################################################
##                       DEFINITION OF VQE HAMILTONIAN                        ##
################################################################################
ExchangeIntegrals = [5.67, 2.32, 3.78]
Ops = [\
        qml.PauliX(0) @ qml.PauliX(1),\
        qml.PauliY(0) @ qml.PauliY(1),\
        qml.PauliZ(0) @ qml.PauliZ(1)\
      ]
VQE_Ham = qml.Hamiltonian(ExchangeIntegrals,Ops)

################################################################################
##                    DIAGONALIZE HAMILTONIAN WITH NUMPY                      ##
################################################################################
## Definition of Pauli Matrices
PauliX = np.array([
    [0,1],
    [1,0]
])
PauliY = np.array([
    [0,-1j],
    [1j,0]
])
PauliZ = np.array([
    [1,0],
    [0,-1]
])
Paulis = [PauliX, PauliY, PauliZ]
## Definition of Hamiltonian
VQE_Ham_Mat = np.sum(\
        ExchangeIntegrals[idx] * np.kron(Paulis[idx], Paulis[idx]) \
        for idx in range(3))
## Diagonalization of Hamiltonian
EigVals, EigVecs = np.linalg.eig(VQE_Ham_Mat)
## Print Ham Matrix and Energies
print('Hamiltonian Matrix: ')
print(VQE_Ham_Mat)
print('Eigenstates: ')
print(EigVecs)
print('Energies: ')
print(EigVals)

################################################################################
##                         DEFINITION OF THE VQE QNN                          ##
################################################################################
dev = qml.device('default.qubit',wires=2)

################################################################################
##                         DEFINITION OF THE VQE QNN                          ##
################################################################################
def VQE_QNN(params, wires):
    '''
    Function for defining
    the VQE QNN
    '''
    for idx in range(2):
        ## CAREFUL! The unpacker * is needed
        ## because of the way qml.Rot works
        qml.Rot(*params[idx],wires=idx)
    qml.CNOT(wires=[0,1])

################################################################################
##                            GET COST FUNCTION                               ##
################################################################################
cost_fn = qml.ExpvalCost(VQE_QNN, VQE_Ham, dev)

################################################################################
##                      INITIALIZE OPTIMIZER AND SEED                         ##
################################################################################
optimizer = qml.GradientDescentOptimizer(stepsize = 0.1)
np.random.seed(0)
## REMEMBER: We are using 2 qubits, and 3 parameters per rotation per qubit
params = np.random.normal(0, np.pi, (2,3))

################################################################################
##                  ITERATE UNTIL CONVERGENCE OF GRAD DESCENT                 ##
################################################################################
MAX_ITER = 200
TOL = 1e-6
for it in range(MAX_ITER):
    ## IMPORTANT! Remember that cost
    ## is computed before optimization step
    params, prev_energy = optimizer.step_and_cost(cost_fn, params)
    vqe_energy = cost_fn(params)
    ## Compute convergence
    dE = np.abs(vqe_energy - prev_energy)
    ## Exit if convergence
    if dE <= TOL:
        print('Optimization converged!')
        print('Iteration {} ; dE = {:.8f}; E = {:.3f}'.format(\
            it,dE,vqe_energy))
        break
    ## Print partial results
    if it%20 == 0:
        print('Iteration {} ; dE = {:.8f}; E = {:.3f}'.format(\
            it,dE,vqe_energy))
## Print final energy
print('VQE Energy: {:.3f}'.format(cost_fn(params)))
