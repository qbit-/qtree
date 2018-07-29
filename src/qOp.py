import numpy as np
import logging as log
import re

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
        if op_identif=='cz':
            return cZ(q_idx)
        if op_identif=='x_1_2':
            return X2(q_idx)
        if op_identif=='y_1_2':
            return Y2(q_idx)
    def __str__(self):
        raise NotImplementedError


class H(qOperation):
    matr = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])

    def __init__(self,qubit):
        if isinstance(qubit,int):
            self.qubit_idx = qubit
    def apply(self,vec):
        return np.dot(self.matr,vec)
    def __str__(self):
        return "<Haramard operator on %s>"%(self.qubit_idx)

class cZ(qOperation):
    #matr = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])

    def __init__(self,qubit):
        if isinstance(qubit,tuple):
            self.qubit_idx = qubit
    def apply(self,vec):
        return np.dot(self.matr,vec)
    def __str__(self):
        return "<control-Z operator on %s>"%(
            str(self.qubit_idx)
        )

class X2(qOperation):
    #matr = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])
    def __init__(self,qubit):
        if isinstance(qubit,tuple):
            self.qubit_idx = qubit
    def apply(self,vec):
        return np.dot(self.matr,vec)
    def __str__(self):
        return "<half-X operator on %s>"%(
            str(self.qubit_idx)
        )
class Y2(qOperation):
    #matr = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])
    def __init__(self,qubit):
        if isinstance(qubit,tuple):
            self.qubit_idx = qubit
    def apply(self,vec):
        return np.dot(self.matr,vec)
    def __str__(self):
        return "<half-Y operator on %s>"%(
            str(self.qubit_idx)
        )
