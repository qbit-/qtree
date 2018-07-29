import numpy as np
import logging as log
import re, cirq

class qOperation:
    def factory(self,arg):
        if isinstance(arg,str):
            return self._create_from_string(arg)

    def _create_from_string(sefl,s):
        log.debug("creating op from '%s'"%s)
        m = re.search(r'(h|t|cz|x_1_2|y_1_2) ([0-9]+)( [0-9]+|)',s)
        if not m:
            raise Exception("file format error in %s"%s)
        op_identif = m.group(1)
        try:
            q_idx = ( int(m.group(2)), int(m.group(3)) )
        except ValueError as e:
            q_idx = int( m.group(2) )

        if op_identif=='h':
            return H(q_idx)
        if op_identif=='t':
            return T(q_idx)
        if op_identif=='cz':
            return cZ(q_idx)
        if op_identif=='x_1_2':
            return X2(q_idx)
        if op_identif=='y_1_2':
            return Y2(q_idx)
    def get_grid_idx(self,grid_size):
        try:
            return [(i//grid_size,i%grid_size) for i in self.qubit_idx]
        except TypeError:
            return [(i//grid_size,i%grid_size) for i in [self.qubit_idx]]
    def to_cirq(self,grid_size):
        return self.cirq_op(
            *[cirq.GridQubit(*x) for x in self.get_grid_idx(grid_size)]
             )

    def __str__(self):
        return "<%s operator on %s>"%(self.name , self.qubit_idx)
    def __repr__(self):
        return self.__str__()


class H(qOperation):
    matr = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])
    name ='H'
    cirq_op = cirq.H
    def __init__(self,qubit):
        if isinstance(qubit,int):
            self.qubit_idx = qubit
    def apply(self,vec):
        return np.dot(self.matr,vec)

class cZ(qOperation):
    #matr = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])
    cirq_op = cirq.CZ
    name = 'cZ'
    def __init__(self,qubit):
        if isinstance(qubit,tuple):
            self.qubit_idx = qubit
    def apply(self,vec):
        return np.dot(self.matr,vec)
    def __str__(self):
        return "<control-Z operator on %s>"%(
            str(self.qubit_idx)
        )

class T(qOperation):
    #matr = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])
    name='T'
    cirq_op = cirq.T
    def __init__(self,qubit):
        if isinstance(qubit,int):
            self.qubit_idx = qubit
    def apply(self,vec):
        return np.dot(self.matr,vec)
    def __str__(self):
        return "<T operator on %s>"%(
            str(self.qubit_idx)
        )

class X2(qOperation):
    #matr = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])
    name='√X'
    cirq_op = lambda s,x: cirq.X(x)**0.5
    def __init__(self,qubit):
        if isinstance(qubit,int):
            self.qubit_idx = qubit
    def apply(self,vec):
        return np.dot(self.matr,vec)
    def __str__(self):
        return "<half-X operator on %s>"%(
            str(self.qubit_idx)
        )
class Y2(qOperation):
    #matr = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])
    name='√Y'
    cirq_op = lambda s,x: cirq.Y(x)**0.5
    def __init__(self,qubit):
        if isinstance(qubit,int):
            self.qubit_idx = qubit
    def apply(self,vec):
        return np.dot(self.matr,vec)
    def __str__(self):
        return "<half-Y operator on %s>"%(
            str(self.qubit_idx)
        )
