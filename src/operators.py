import numpy as np
import re
import cirq
import logging
log = logging.getLogger('qtree')


class qOperation:
    def factory(self, arg):
        if isinstance(arg, str):
            return self._create_from_string(arg)

    def _create_from_string(sefl, s):
        #log.debug("creating op from '{}'".format(s))
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
    matrix = 1/np.sqrt(2) * np.array([[ 1.+0.j,  1.+0.j],
                                      [ 1.+0.j, -1.+0.j]])
    name = 'H'
    cirq_op = cirq.H
    diagonal = False
    n_qubit = 1

    cirq_op = cirq.H
    
    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits
        self.tensor = self.matrix

    def apply(self, vec):
        return np.dot(self.matr, vec)


class cZ(qOperation):
    matrix = np.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                       [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
                       [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
                       [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]])
    name = 'cZ'
    diagonal = True
    n_qubit = 2

    cirq_op = cirq.CZ

    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits
        # indexes are (in1,out1,in2,out2)
        self.tensor = np.array(
            [
                [
                    [[1.+0j, 0+0j],
                     [0.+0j, 1+0j]],
                    [[0.+0j, 0+0j],
                     [0.+0j, 0+0j]]
                ],
                [
                    [[0.+0j, 0+0j],
                     [0.+0j, 0+0j]],
                    [[1.+0j, 0+0j],
                     [0.+0j, -1+0j]]
                ]
            ]
        )
        self.tensor = self.get_simplified()

    def get_simplified(self):
        t = self.tensor
        r = np.arange(t.shape[0])
        # cZ_ijkl (diagonal)-> cZ_iijj -> cz_ij
        return np.array([[t[i,i,j,j] for i in r] for j in r])

    def apply(self, vec):
        return np.dot(self.matr, vec)


class T(qOperation):
    matrix = np.array([[1.+0.j,  0.+0.j        ],
                       [0.+0.j, np.exp(1.j*np.pi/4)]])
    name = 'T'
    n_qubit = 1

    cirq_op = cirq.T
    diagonal = True

    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits
        self.tensor = self.matrix
        self.tensor = self.get_simplified()

    def apply(self, vec):
        return np.dot(self.matr, vec)

    def get_simplified(self):
	# cZ_ijkl (diagonal)-> cZ_iijj -> cz_ij
        return self.tensor.diagonal()

class X_1_2(qOperation):
    matrix = np.array([[0.5+0.5j, 0.5-0.5j],
                       [0.5-0.5j, 0.5+0.5j]])
    name = 'X_1_2'
    diagonal = False
    n_qubit = 1

    def cirq_op(self, x): return cirq.X(x)**0.5

    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits
        self.tensor = self.matrix

    def apply(self, vec):
        return np.dot(self.matr, vec)


class Y_1_2(qOperation):
    matrix = np.array([[ 0.5+0.5j, -0.5-0.5j],
                       [ 0.5+0.5j,  0.5+0.5j]])
    name = 'Y_1_2'
    diagonal = False
    n_qubit = 1

    def cirq_op(self, x): return cirq.Y(x)**0.5

    def __init__(self, *qubits):
        self._check_qubit_count(qubits)
        self._qubits = qubits
        self.tensor = self.matrix

    def apply(self, vec):
        return np.dot(self.matr, vec)

