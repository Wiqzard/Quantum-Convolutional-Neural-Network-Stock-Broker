import torch
import itertools
import qiskit

# from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit.circuit.random import random_circuit
import numpy as np


class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
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

        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)

        return np.array([expectation])


class QuanvCircuit:
    """
    This class defines filter circuit of Quanvolution layer
    """

    def __init__(self, kernel_size, backend, shots, threshold):
        # --- Circuit definition start ---
        self.n_qubits = kernel_size**2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]

        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)

        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        # ---- Circuit definition end ----

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        data = torch.reshape(data, (1, self.n_qubits))
        thetas = []
        for dat in data:
            theta = []
            for val in dat:
                if val > self.threshold:
                    theta.append(np.pi)
                else:
                    theta.append(0)
            thetas.append(theta)

        param_dict = {
            self.theta[i]: theta[i]
            for theta, i in itertools.product(thetas, range(self.n_qubits))
        }

        param_binds = [param_dict]

        # execute random quantum circuit
        job = qiskit.execute(
            self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds
        )
        result = job.result().get_counts(self._circuit)

        # decoding the result
        counts = sum(
            sum(int(char) for char in key) * val for key, val in result.items()
        )

        # Compute probabilities for each state
        probabilities = counts / (self.shots * self.n_qubits)
        # probabilities = counts / self.shots
        return probabilities
