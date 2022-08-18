from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, assemble
from qiskit.quantum_info import Statevector
import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit.algorithms.linear_solvers.hhl import HHL


def inner_prod(vec1, vec2):
    # first check if lengths are equal
    if len(vec1) != len(vec2):
        raise ValueError('Lengths of states are not equal')
    N = len(vec1)
    nqubits = math.ceil(np.log2(N))

    # normalizing
    vec1norm = np.linalg.norm(vec1)
    vec2norm = np.linalg.norm(vec2)
    vec1 = vec1 / vec1norm
    vec2 = vec2 / vec2norm

    circ = QuantumCircuit(nqubits + 1)
    vec = np.concatenate((vec1, vec2)) / np.sqrt(2)

    circ.initialize(vec, range(nqubits + 1))
    circ.h(nqubits)

    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend, backend_options={"zero_threshold": 1e-20})

    result = job.result()
    o = np.real(result.get_statevector(circ))

    m_sum = 0
    for l in range(N):
        m_sum += o[l]**2

    return (2 * m_sum - 1) * vec1norm * vec2norm


def multiply(A, B):
    result = np.zeros(shape=(A.shape[0], B.shape[1]))
    for i, row in enumerate(A):
        for j, col in enumerate(B.T):
            result[i][j] = inner_prod(row, col)

    return result


def solve_linear_equation(X, b):
    norm = np.linalg.norm(b)
    b = b / norm
    offset = 0
    if X.shape != X.T.shape or (X.T - X).any():
        X, b = make_symetric(X, b)
        offset = len(X) // 2
    naive_hhl_solution = HHL().solve(X, b)
    naive_sv = Statevector(naive_hhl_solution.state).data
    naive_full_vector = np.real(
        naive_sv[len(naive_sv) // 2:len(naive_sv) // 2 + len(X)])
    return (norm * naive_hhl_solution.euclidean_norm *
            naive_full_vector / np.linalg.norm(naive_full_vector))[offset:]


def make_symetric(X, b):
    #TODO Postaraj się, aby działała dla macierzy prostokątnych
    n = X.shape[0]
    m = X.shape[1]
    C = np.zeros(shape=(2 * n, 2 * m))
    C[0:n, m:2 * m] = X
    C[m:2 * m, 0:n] = X.T
    d = np.zeros(shape=(2 * len(b)))
    d[0:n] = b
    return C, d
