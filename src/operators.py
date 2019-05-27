"""
This module implements quantum gates from the CMON set of Google
"""
import numpy as np
import re
import cirq

from src.logger_setup import log
from math import sqrt, pi
from cmath import exp

import src.system_defs as defs


class Gate:
    """
    Base class for quantum gates.
    """
    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits

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
        return "{}({})".format(type(self).__name__,
                               ','.join(map(str, self._qubits)))

    def __repr__(self):
        return self.__str__()


class H(Gate):
    """
    Hadamard gate
    """
    tensor = 1/sqrt(2) * np.array([[1j,  1j],
                                   [1j, -1j]],
                                  dtype=defs.NP_ARRAY_TYPE)
    cirq_op = cirq.H
    diagonal = False
    n_qubit = 1

    cirq_op = cirq.H


class cZ(Gate):
    """
    Controlled :math:`Z` gate
    """
    tensor = np.array([[1, 1],
                       [1, -1]], dtype=defs.NP_ARRAY_TYPE)

    diagonal = True
    n_qubit = 2

    cirq_op = cirq.CZ


class T(Gate):
    """
    :math:`T`-gate
    """
    tensor = np.array([exp(-1.j*pi/8), exp(1.j*pi/8)],
                      dtype=defs.NP_ARRAY_TYPE)
    n_qubit = 1

    cirq_op = cirq.T
    diagonal = True


class S(Gate):
    """
    :math:`S`-gate
    """
    tensor = np.array([exp(-1.j*pi/4), exp(1.j*pi/4)],
                      dtype=defs.NP_ARRAY_TYPE)
    n_qubit = 1

    cirq_op = cirq.T
    diagonal = True


class X_1_2(Gate):
    """
    :math:`X^{1/2}`
    gate
    """
    tensor = 1/sqrt(2) * np.array([[1, 1j],
                                   [1j, 1]],
                                  dtype=defs.NP_ARRAY_TYPE)
    diagonal = False
    n_qubit = 1

    def cirq_op(self, x): return cirq.X(x)**0.5


class Y_1_2(Gate):
    r"""
    :math:`Y^{1/2}` gate
    """
    tensor = 1/sqrt(2) * np.array([[1, -1],
                                   [1,  1]],
                                  dtype=defs.NP_ARRAY_TYPE)
    diagonal = False
    n_qubit = 1

    def cirq_op(self, x): return cirq.Y(x)**0.5


class X(Gate):
    tensor = np.array([[0.+0.j, 1.+0j],
                       [1.+0j, 0.+0j]],
                      dtype=defs.NP_ARRAY_TYPE)
    diagonal = False
    n_qubit = 1

    def cirq_op(self, x): return cirq.X(x)


# class cX(Gate):
#     raise NotImplemented
#     diagonal = False
#     n_qubit = 1

#     def cirq_op(self, x): raise NotImplemented('No cX operation in Cirq')


class Y(Gate):
    tensor = np.array([[0.-1j, 0.+0j],
                       [0.+0j, 0.+1j]],
                      dtype=defs.NP_ARRAY_TYPE)
    diagonal = False
    n_qubit = 1

    def cirq_op(self, x): return cirq.Y(x)


class ZP(Gate):
    """Arbitrary :math:`Z` rotation"""

    def __init__(self, *qubits, alpha):
        super().__init__(*qubits)
        self._alpha = alpha
        self.tensor = np.array([1, exp(1.j*alpha)],
                               dtype=defs.NP_ARRAY_TYPE)

    def __str__(self):
        return "{}[a={:.2f}]({})".format(type(self).__name__,
                                         self._alpha,
                                         ','.join(map(str, self._qubits)))
    n_qubit = 1
    cirq_op = cirq.T
    diagonal = True


def read_circuit_file(filename, max_depth=None):
    """
    Read circuit file and return quantum circuit in the
    form of a list of lists

    Parameters
    ----------
    filename : str
             circuit file in the format of Sergio Boixo
    max_depth : int
             maximal depth of gates to read

    Returns
    -------
    qubit_count : int
            number of qubits in the circuit
    circuit : list of lists
            quantum circuit as a list of layers of gates
    """
    label_to_gate_dict = {
        'h': H,
        't': T,
        'cz': cZ,
        'x_1_2': X_1_2,
        'y_1_2': Y_1_2,
    }

    log.info("reading file {}".format(filename))
    circuit = []
    circuit_layer = []

    with open(filename, "r") as fp:
        qubit_count = int(fp.readline())
        log.info("There are {:d} qubits in circuit".format(qubit_count))
        n_ignored_layers = 0
        current_layer = 0

        for idx, line in enumerate(fp):
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
            m = re.search(r'(?P<operation>h|t|cz|x_1_2|y_1_2) (?P<qubit1>[0-9]+) ?(?P<qubit2>[0-9]+)?', op_str)
            if m is None:
                raise Exception("file format error in {}".format(s))

            op_identif = m.group('operation')

            if m.group('qubit2') is not None:
                q_idx = (int(m.group('qubit1')), int(m.group('qubit2')))
            else:
                q_idx = (int(m.group('qubit1')),)

            op = label_to_gate_dict[op_identif](*q_idx)
            circuit_layer.append(op)

        circuit.append(circuit_layer)  # last layer

        if n_ignored_layers > 0:
            log.info("Ignored {} layers".format(n_ignored_layers))

    return qubit_count, circuit


# Dictionary containing data of all operators
# in this module. Only nonzero entries of tensors are listed, which means
# diagonals of diagonal matrices or double diagonal for diagonal
# fourth order tensors
operator_values_dict = {
    'h': H(1).tensor,
    'x_1_2': X_1_2(1).tensor,
    'y_1_2': Y_1_2(1).tensor,
    't': T(1).tensor,
    'cz': cZ(1, 1).tensor
}
