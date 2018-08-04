import numpy as np
import logging as log
import re
import cirq
from math import sqrt, pi
from cmath import exp

class qOperation:
    def factory(self, arg):
        if isinstance(arg, str):
            return self._create_from_string(arg)

    def _create_from_string(sefl, s):
        log.debug("creating op from '{}'".format(s))
        m = re.search(
            r'(?P<operation>h|t|cz|x_1_2|y_1_2) (?P<qubit1>[0-9]+) ?(?P<qubit2>[0-9]+)?', s)
        if m is None:
            raise Exception("file format error in {}".format(s))
        op_identif = m.group('operation')

        if m.group('qubit2') is not None:
            q_idx = int(m.group('qubit1')), int(m.group('qubit2'))
        else:
            q_idx = int(m.group('qubit1'))

        if op_identif == 'h':
            return H(q_idx)
        if op_identif == 't':
            return T(q_idx)
        if op_identif == 'cz':
            return cZ(*q_idx)
        if op_identif == 'x_1_2':
            return X_1_2(q_idx)
        if op_identif == 'y_1_2':
            return Y_1_2(q_idx)

    def _check_qubit_count(self, qubits):
        if len(qubits) != self.n_qubit:
            raise ValueError(
                "Wrong number of qubits: {}, required: {}".format(
                    len(qubits), self.n_qubit))

    def to_cirq_2d_circ_op(self, side_length):
        return self.cirq_op(
            *[cirq.GridQubit(*np.unravel_index(
                qubit, [side_length, side_length]))
              for qubit in self._qubits]
        )

    def __str__(self):
        return "<{} operator on {}>".format(self.name, self._qubits)

    def __repr__(self):
        return self.__str__()

        
class H(qOperation):
    matrix = 1/sqrt(2) * np.array([[ -1j,  -1j],
                                   [ -1j, 1j]])
    # matrix = -1j * matrix
    name = 'H'
    cirq_op = cirq.H
    diagonal = False
    n_qubit = 1

    cirq_op = cirq.H
    
    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits

    def apply(self, vec):
        return np.dot(self.matr, vec)


class cZ(qOperation):
    matrix = np.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                       [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
                       [ 0.+0.j,  0.+0.j,  1,  0.+0.j],
                       [ 0.+0.j,  0.+0.j,  0.+0.j, -1]])
    name = 'cZ'
    diagonal = True
    n_qubit = 2

    cirq_op = cirq.CZ

    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits

    def apply(self, vec):
        return np.dot(self.matr, vec)


class T(qOperation):
    matrix = np.array([[exp(-1.j*pi/8),  0],
                       [0,  exp(1.j*pi/8)]])
    name = 'T'
    n_qubit = 1

    cirq_op = cirq.T
    diagonal = True

    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits

    def apply(self, vec):
        return np.dot(self.matr, vec)


class X_1_2(qOperation):
    matrix = 1/sqrt(2) * np.array([[1, -1j],
                                   [-1j, 1]])
    name = 'X_1_2'
    diagonal = False
    n_qubit = 1

    def cirq_op(self, x): return cirq.X(x)**0.5

    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits

    def apply(self, vec):
        return np.dot(self.matr, vec)


class Y_1_2(qOperation):
    matrix = 1/sqrt(2) * np.array([[ 1, 1],
                                   [ -1,  1]])
    name = 'Y_1_2'
    diagonal = False
    n_qubit = 1

    def cirq_op(self, x): return cirq.Y(x)**0.5

    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits

    def apply(self, vec):
        return np.dot(self.matr, vec)


def read_circuit_file(filename, max_depth=None):
    log.info("reading file {}".format(filename))
    circuit = []
    circuit_layer = []
    
    with open(filename, "r") as fp:
        qubit_count = int(fp.readline())
        log.info("There are {:d} qubits in circuit".format(qubit_count))
        n_ignored_layers = 0
        current_layer = 0

        for line in fp:
            m = re.search(r'(?P<layer>[0-9]+) (?=[a-z])', line)
            if m is None:
                raise Exception("file format error at line {}".format(idx))
            # Read circuit layer by layer
            layer_num = int(m.group('layer'))

            if max_depth is not None and layer_num > max_depth:
                n_ignored_layers = layer_num - max_depth
                continue

            if layer_num > current_layer:
                circuit.append(circuit_layer)
                circuit_layer = []
                current_layer = layer_num

            op_str = line[m.end():]
            op = qOperation().factory(op_str)
            circuit_layer.append(op)

        circuit.append(circuit_layer) # last layer

        if n_ignored_layers > 0:
            log.info("Ignored {} layers".format(n_ignored_layers))

    return qubit_count, circuit
