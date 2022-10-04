import torch
import itertools
import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit.circuit.random import random_circuit
from qiskit.circuit import Parameter


class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = qiskit.transpile(self._circuit, self.backend)
        qobj = qiskit.assemble(
            t_qc,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)

        return np.array([expectation])


# @title QCircuit
class QuantumVCircuit:
    def __init__(self, kernel_size, backend, shots, threshold):
        """
        quantum circuit class to execute the convolution operation
        args:
            kernel_size   =   size of (same as classical conv) kernel/ the image dimension patch
            backend       =   the quantum computer hardware to use (only simulator is used here)
            shots         =   how many times to run the circuit to get a probability distribution
            threshold     =   the threshold value (0-255) to assign theta
        """
        self.n_qubits = kernel_size**2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self._params = [Parameter(f"θ_{i}") for i in range(self.n_qubits)]

        for i in range(self.n_qubits):
            self._circuit.rx(self._params[i], i)

        self._circuit.barrier()
        # add unitary random circuit
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        # reshape input data-> [1, kernel_size, kernel_size] -> [1, self.n_qubits]
        data = torch.reshape(data, (1, self.n_qubits))
        # encoding data to parameters
        thetas = []
        for dat in data:
            theta = []
            for val in dat:
                if val > self.threshold:
                    theta.append(np.pi)
                else:
                    theta.append(0)
            thetas.append(theta)
        # for binding parameters
        param_dict = {
            self._params[i]: theta[i]
            for theta, i in itertools.product(thetas, range(self.n_qubits))
        }

        param_binds = [param_dict]

        # execute random quantum circuit
        result = (
            qiskit.execute(
                self._circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            .result()
            .get_counts()
        )

        # decoding the result
        counts = sum(
            sum(int(char) for char in key) * val for key, val in result.items()
        )

        # Compute probabilities for each state
        probabilities = counts / (self.shots * self.n_qubits)
        return probabilities
