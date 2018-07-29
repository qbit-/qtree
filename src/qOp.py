import numpy as np
import logging as log

class qOperation:
    def __init__(self,arg):
        if isinstance(arg,str):
            log.debug("creating op from %s"%arg)
            self._create_from_string(arg)

    def _create_from_string(sefl,s):
        raise NotImplementedError

class H(qOperation):

    matr = 1/np.sqrt(2)* np.array([[1,1],[1,-1]])

    def __init__(self,qubit):
        if isinstance(qubit,int):
            self.qubit_idx = qubit

    def apply(self,vec):
        return np.dot(self.matr,vec)

    def __repr__(self):
        return "<Haramard operator on %s>"%(self.qubit_idx)
