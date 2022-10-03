from qiskit import (
    QuantumCircuit,
    Aer,
    IBMQ,
    QuamtumRegister,
    ClassicalRegister,
    QuantumRegister,
)


q = QuantumRegister(N)
c = ClassicalRegister(1)
qc = QuantumCircuit(q, c)

qc.draw("mpl")
