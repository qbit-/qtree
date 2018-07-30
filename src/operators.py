import numpy as np
import logging as log
import re
import cirq


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
            q_idx = (int(m.group('qubit1')), int(m.group('qubit2')))
        else:
            q_idx = int(m.group('qubit1'))

        if op_identif == 'h':
            return H(q_idx)
        if op_identif == 't':
            return T(q_idx)
        if op_identif == 'cz':
            return cZ(q_idx)
        if op_identif == 'x_1_2':
            return X_1_2(q_idx)
        if op_identif == 'y_1_2':
            return Y_1_2(q_idx)

    def get_grid_idx(self, grid_size):
        try:
            return [(i//grid_size, i % grid_size) for i in self.qubit_idx]
        except TypeError:
            return [(i//grid_size, i % grid_size) for i in [self.qubit_idx]]

    def to_cirq(self, grid_size):
        return self.cirq_op(
            *[cirq.GridQubit(*x) for x in self.get_grid_idx(grid_size)]
        )

    def __str__(self):
        return "<{} operator on {}>".format(self.name, self.qubit_idx)

    def __repr__(self):
        return self.__str__()


class H(qOperation):
    matrix = 1/np.sqrt(2) * np.array([[ 1.+0.j,  1.+0.j],
                                      [ 1.+0.j, -1.+0.j]])
    name = 'H'
    cirq_op = cirq.H
    diagonal = False

    def __init__(self, qubit):
        if isinstance(qubit, int):
            self.qubit_idx = qubit

    def apply(self, vec):
        return np.dot(self.matr, vec)


class cZ(qOperation):
    matrix = np.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                       [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
                       [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
                       [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]])
    cirq_op = cirq.CZ
    name = 'cZ'
    diagonal = True

    def __init__(self, qubit):
        if isinstance(qubit, tuple):
            self.qubit_idx = qubit

    def apply(self, vec):
        return np.dot(self.matr, vec)


class T(qOperation):
    matrix = np.array([[1.+0.j,  0.+0.j        ],
                       [0.+0.j, np.exp(1.j*np.pi/4)]])
    name = 'T'
    cirq_op = cirq.T
    diagonal = True

    def __init__(self, qubit):
        if isinstance(qubit, int):
            self.qubit_idx = qubit

    def apply(self, vec):
        return np.dot(self.matr, vec)


class X_1_2(qOperation):
    matrix = np.array([[0.5+0.5j, 0.5-0.5j],
                       [0.5-0.5j, 0.5+0.5j]])
    name = 'X_1_2'
    diagonal = False

    def cirq_op(s, x): return cirq.X(x)**0.5

    def __init__(self, qubit):
        if isinstance(qubit, int):
            self.qubit_idx = qubit

    def apply(self, vec):
        return np.dot(self.matr, vec)


class Y_1_2(qOperation):
    matrix = np.array([[ 0.5+0.5j, -0.5-0.5j],
                       [ 0.5+0.5j,  0.5+0.5j]])
    name = 'Y_1_2'
    diagonal = False

    def cirq_op(self, x): return cirq.Y(x)**0.5

    def __init__(self, qubit):
        if isinstance(qubit, int):
            self.qubit_idx = qubit

    def apply(self, vec):
        return np.dot(self.matr, vec)


