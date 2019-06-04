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
    Properties:
    ----------
    name: str
            The name of the gate
    tensor: numpy.array
            The gate tensor. For each qubit a gate
            either introduces a new variable (non-diagonal gate, like X)
            or does not (diagonal gate, like T). Multiqubit gates
            can be diagonal on some of the variables, and not diagonal on
            others (like ccX). The order of dimensions IS ALWAYS
            (a, new_a, b, new_b, c, d, new_d, ...)

    qubits: tuple
            Qubits the gate acts on

    changed_qubits : tuple
            Tuple of ints which states what qubit's bases are changed
            (along which qubits the gate is not diagonal).

    cirq_op: Cirq.GridQubit
            Cirq 2D gate. Used for unit tests. Optional

    data_hash: int
             hash of the gate's tensor. Used to store all gate
             tensors separately from their identifiers in the code
    """
    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = tuple(qubits)
        self._data_hash = hash(self.tensor.tobytes())

    def _check_qubit_count(self, qubits):
        n_qubits = len(self.tensor.shape) - len(self._changes_qubits)
        if len(qubits) != n_qubits:
            raise ValueError(
                "Wrong number of qubits: {}, required: {}".format(
                    len(qubits), n_qubits))

    @property
    def name(self):
        return type(self).__name__

    @property
    def qubits(self):
        return self._qubits

    @property
    def data_hash(self):
        return self._data_hash

    @property
    def changed_qubits(self):
        return tuple(self._qubits[idx] for idx in self._changes_qubits)

    def to_cirq_2d_circ_op(self, side_length):
        return self.cirq_op(
            *[cirq.GridQubit(*np.unravel_index(
                qubit, [side_length, side_length]))
              for qubit in self._qubits]
        )

    def __str__(self):
        return "{}({})".format(self.name,
                               ','.join(map(str, self._qubits)))

    def __repr__(self):
        return self.__str__()


class I(Gate):
    tensor = np.array([1,  1], dtype=defs.NP_ARRAY_TYPE)
    cirq_op = cirq.I
    _changes_qubits = tuple()


class H(Gate):
    """
    Hadamard gate
    """
    tensor = 1/sqrt(2) * np.array([[1,  1],
                                   [1, -1]],
                                  dtype=defs.NP_ARRAY_TYPE)
    cirq_op = cirq.H
    _changes_qubits = (0, )


class cZ(Gate):
    """
    Controlled :math:`Z` gate
    """
    tensor = np.array([[1, 1],
                       [1, -1]], dtype=defs.NP_ARRAY_TYPE)
    cirq_op = cirq.CZ
    _changes_qubits = tuple()


class Z(Gate):
    """
    :math:`Z`-gate
    """
    tensor = np.array([1, -1],
                      dtype=defs.NP_ARRAY_TYPE)
    cirq_op = cirq.Z
    _changes_qubits = tuple()


class T(Gate):
    """
    :math:`T`-gate
    """
    tensor = np.array([1, exp(1.j*pi/4)],
                      dtype=defs.NP_ARRAY_TYPE)
    cirq_op = cirq.T
    _changes_qubits = tuple()


class S(Gate):
    """
    :math:`S`-gate
    """
    tensor = np.array([1, exp(1.j*pi/2)],
                      dtype=defs.NP_ARRAY_TYPE)
    cirq_op = cirq.S
    _changes_qubits = tuple()


class X_1_2(Gate):
    """
    :math:`X^{1/2}`
    gate
    """
    tensor = 1/2 * np.array([[1 + 1j, 1 - 1j],
                             [1 - 1j, 1 + 1j]],
                            dtype=defs.NP_ARRAY_TYPE)

    def cirq_op(self, x): return cirq.X(x)**0.5
    _changes_qubits = (0, )


class Y_1_2(Gate):
    r"""
    :math:`Y^{1/2}` gate
    """
    tensor = 1/2 * np.array([[1 + 1j, -1 - 1j],
                             [1 + 1j, 1 + 1j]],
                            dtype=defs.NP_ARRAY_TYPE)

    def cirq_op(self, x): return cirq.Y(x)**0.5
    _changes_qubits = (0, )


class X(Gate):
    tensor = np.array([[0, 1],
                       [1, 0]],
                      dtype=defs.NP_ARRAY_TYPE)

    def cirq_op(self, x): return cirq.X(x)
    _changes_qubits = (0, )


# class cX(Gate):
#     raise NotImplemented
#     diagonal = False
#     n_qubit = 1

#     def cirq_op(self, x): raise NotImplemented('No cX operation in Cirq')


class Y(Gate):
    tensor = np.array([[0, -1j],
                       [1j, 0]],
                      dtype=defs.NP_ARRAY_TYPE)

    def cirq_op(self, x): return cirq.Y(x)
    _changes_qubits = (0, )


class ZPhase(Gate):
    """Arbitrary :math:`Z` rotation"""

    def __init__(self, *qubits, alpha):
        super().__init__(*qubits)
        self._alpha = alpha
        self.tensor = np.array([1, exp(1.j*alpha*pi)],
                               dtype=defs.NP_ARRAY_TYPE)

    def __str__(self):
        return "{}[a={:.2f}]({})".format(type(self).__name__,
                                         self._alpha,
                                         ','.join(map(str, self._qubits)))
    _changes_qubits = tuple()


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
        'i': I,
        'h': H,
        't': T,
        'z': Z,
        'cz': cZ,
        'x': X,
        'y': Y,
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
            m = re.search(r'(?P<operation>h|t|z|cz|x|y|x_1_2|i|y_1_2) (?P<qubit1>[0-9]+) ?(?P<qubit2>[0-9]+)?', op_str)
            if m is None:
                raise Exception("file format error in {}".format(op_str))

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
