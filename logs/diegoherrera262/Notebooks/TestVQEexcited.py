################################################################################
##                    EXPLORATION OF EXCITED STATES SEARCH                    ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 29/04/21
## DESCRIPTION: Here I will try to combine two costs functions that appear on
##              the process of finding excited states of a Hamiltonian:
##              1. The Hamiltonian itself
##              2. The overlap with previous eigenstates
##              By now, I will try to combine the two cost functions

################################################################################
##                            NECESSARY IMPORTS                               ##
################################################################################
import pennylane as qml
from pennylane import numpy as np

################################################################################
##                       DEFINITION OF VQE HAMILTONIAN                        ##
################################################################################
ExchangeIntegrals = [1.56, 4.32, 5.78]
Ops = [\
     qml.PauliX(0) @ qml.PauliX(1),\
     qml.PauliY(0) @ qml.PauliY(1),\
     qml.PauliZ(0) @ qml.PauliZ(1)\
   ]
VQE_Ham = qml.Hamiltonian(ExchangeIntegrals,Ops)
Overlap_Ham = qml.Hamiltonian([0,0,1],Ops)

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
##                       DEFINITION OF QUANTUM DEVICE                         ##
################################################################################
dev = qml.device('qiskit.aer',wires=2,backend='qasm_simulator')
statevec_dev = qml.device('qiskit.aer',wires=2,backend='statevector_simulator')

################################################################################
##                         DEFINITION OF THE VQE QNN                          ##
################################################################################
def VQE_QNN(params, **kwargs):
    '''
    Function for defining
    the VQE QNN
    '''
    for idx in range(2):
        ## CAREFUL! The unpacker * is needed
        ## because of the way qml.Rot works
        qml.Rot(*params[idx],wires=idx)
    qml.CNOT(wires=[0,1])

def VQE_QNN_func(params, **kwargs):
    '''
    Function for defining
    the VQE QNN Function
    '''
    VQE_QNN(params, **kwargs)
    return qml.expval(qml.PauliZ(0))

def get_projector_Hamiltonian(params, **kwargs):
    '''
    Function for finding the
    projector that updates
    the Hamiltonian
    '''
    myqnode = qml.QNode(VQE_QNN_func, statevec_dev)
    myqnode(params, **kwargs)
    eigstate = statevec_dev.state
    Proj = np.outer(eigstate.T.conjugate(),eigstate)
    Const_Ham = qml.Hermitian(Proj,wires=[0,1])
    return qml.Hamiltonian([30],[Const_Ham])

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

################################################################################
##                             UPDATE HAMILTONIAN                             ##
################################################################################
VQE_Ham = VQE_Ham + get_projector_Hamiltonian(params)

################################################################################
##                            UPDATE COST FUNCTION                            ##
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
        print('Iteration {} ; dE = {:.8f}; E1 = {:.3f}'.format(\
            it,dE,vqe_energy))
        break
    ## Print partial results
    if it%20 == 0:
        print('Iteration {} ; dE = {:.8f}; E = {:.3f}'.format(\
            it,dE,vqe_energy))
## Print final energy
print('VQE Energy (Exc. 1): {:.3f}'.format(cost_fn(params)))
