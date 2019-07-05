"""
This module implements quantum gates from the CMON set of Google
"""
import numpy as np
import re
import itertools
import cirq

from fractions import Fraction
from qtree.logger_setup import log

import qtree.system_defs as defs


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
            (new_a, a, b_new, b, c, d_new, d, ...)

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
        # supposedly unique id for a class
        self._data_key = hash((self.name, id(self.__class__)))

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
    def data_key(self):
        return self._data_key

    @property
    def changed_qubits(self):
        return tuple(self._qubits[idx] for idx in self._changes_qubits)

    def to_cirq_2d_circ_op(self, side_length):
        return self.cirq_op(
            *[cirq.GridQubit(*np.unravel_index(
                qubit, [side_length, side_length]))
              for qubit in self._qubits]
        )

    def to_cirq_1d_circ_op(self):
        return self.cirq_op(
            *[cirq.LineQubit(qubit) for qubit in self._qubits]
        )

    def __str__(self):
        return "{}({})".format(self.name,
                               ','.join(map(str, self._qubits)))

    def __repr__(self):
        return self.__str__()


class ParametricGate(Gate):
    """
    Gate that may have parameters

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
            (new_a, a, b_new, b, c, d_new, d, ...)

    qubits: tuple
            Qubits the gate acts on

    changed_qubits : tuple
            Tuple of ints which states what qubit's bases are changed
            (along which qubits the gate is not diagonal).

    cirq_op: Cirq.GridQubit
            Cirq 2D gate. Used for unit tests. Optional

    data_key: int
             Unique identifier of the gate's data. Data of
             tensors is stored separately from their identifiers
    parameters: dict
             Parameters used by the gate
    """
    def __init__(self, *qubits, **parameters):
        self._qubits = tuple(qubits)
        # supposedly unique id for an instance
        self._data_key = hash((self.name, id(self)))
        self._create_tensor(**parameters)
        self._check_qubit_count(qubits)

    @property
    def parameters(self):
        return self._parameters

    def __str__(self):
        return ("{}".format(type(self).__name__) +
                "[" + ",".join("{}={:.2f}".format(
                    param_name, float(param_value)) for
                               param_name, param_value in
                               sorted(self._parameters.items(),
                                      key=lambda pair: pair[0]))
                + "]({})".format(','.join(map(str, self._qubits))))


class M(Gate):
    """
    Measurement gate. This is essentially the identity operator, but
    it forces the inntroduction of a variable in the graphical model
    """
    tensor = np.array([[1, 0], [0, 1]], dtype=defs.NP_ARRAY_TYPE)
    cirq_op = cirq.I
    _changes_qubits = (0, )


class I(Gate):
    tensor = np.array([1, 1], dtype=defs.NP_ARRAY_TYPE)
    cirq_op = cirq.I
    _changes_qubits = tuple()


class H(Gate):
    """
    Hadamard gate
    """
    tensor = 1/np.sqrt(2) * np.array([[1,  1],
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
    tensor = np.array([1, np.exp(1.j*np.pi/4)],
                      dtype=defs.NP_ARRAY_TYPE)
    cirq_op = cirq.T
    _changes_qubits = tuple()


class X_1_2(Gate):
    """
    :math:`X^{1/2}`
    gate
    """
    tensor = Fraction(1, 2) * np.array([[1 + 1j, 1 - 1j],
                                        [1 - 1j, 1 + 1j]],
                                       dtype=defs.NP_ARRAY_TYPE)

    def cirq_op(self, x): return cirq.X(x)**0.5
    _changes_qubits = (0, )


class Y_1_2(Gate):
    r"""
    :math:`Y^{1/2}` gate
    """
    tensor = Fraction(1, 2) * np.array([[1 + 1j, -1 - 1j],
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


class cX(Gate):
    tensor = np.array([[[1., 0.],
                        [0., 1.]],
                       [[0., 1.],
                        [1., 0.]]])
    _changes_qubits = (1, )
    cirq_op = cirq.CNOT


class Y(Gate):
    tensor = np.array([[0, -1j],
                       [1j, 0]],
                      dtype=defs.NP_ARRAY_TYPE)

    def cirq_op(self, x): return cirq.Y(x)
    _changes_qubits = (0, )


class ZPhase(ParametricGate):
    """Arbitrary :math:`Z` rotation
    [[1, 0],
    [0, g]],  where

    g = exp(i·π·t)
    """

    _changes_qubits = tuple()

    def _create_tensor(self, alpha=1):
        """Rotation along Z axis"""
        self.tensor = np.array([1., np.exp(1j * np.pi * alpha)])
        self._parameters = {'alpha': alpha}

    def cirq_op(self, x): return cirq.ZPowGate(
            exponent=float(self._parameters['alpha']))(x)


class XPhase(ParametricGate):
    """Arbitrary :math:`X` rotation
    [[g·c, -i·g·s],
    [-i·g·s, g·c]], where

    c = cos(π·alpha/2), s = sin(π·alpha/2), g = exp(i·π·alpha/2).
    """

    _changes_qubits = (0, )

    def _create_tensor(self, alpha=1):
        """Rotation along X axis"""
        c = np.cos(np.pi*alpha/2)
        s = np.sin(np.pi*alpha/2)
        g = np.exp(1j*np.pi*alpha/2)

        self.tensor = np.array([[g*c, -1j*g*s],
                                [-1j*g*s, g*c]])

        self._parameters = {'alpha': alpha}

    def cirq_op(self, x): return cirq.XPowGate(
            exponent=float(self._parameters['alpha']))(x)


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

    operation_search_patt = r'(?P<operation>' + r'|'.join(label_to_gate_dict.keys()) + r')(?P<qubits>( \d+)+)'

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
            m = re.search(operation_search_patt, op_str)
            if m is None:
                raise Exception("file format error in {}".format(op_str))

            op_identif = m.group('operation')

            q_idx = tuple(int(qq) for qq in m.group('qubits').split())

            op = label_to_gate_dict[op_identif](*q_idx)
            circuit_layer.append(op)

        circuit.append(circuit_layer)  # last layer

        if n_ignored_layers > 0:
            log.info("Ignored {} layers".format(n_ignored_layers))

    return qubit_count, circuit
