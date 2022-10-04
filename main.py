from qiskit import (
    QuantumCircuit,
    Aer,
    IBMQ,
    ClassicalRegister,
    QuantumRegister,
)

N = 4
q = QuantumRegister(N)
c = ClassicalRegister(1)
qc = QuantumCircuit(q, c)

qc.draw("mpl")


#
"""
1. Translate classical states(e.g. dataset points) to qunatum states.
    -> into expectation of qubit
    -> x_1,...x_n d dimensional vectors -> 
"""
