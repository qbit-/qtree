import numpy as np
import logging as log

class Tensor():
    def __init__(self,op=None):
        if op:
            if op.name=='cZ':
                self._tensor = np.sum(
                    op.tensor,
                    axis=(0,2))
            else:
                self._tensor = op.tensor
            #self._op = op
        self.variables = []

    def add_variable(self,*vs):
        log.debug('adding vars'+str(vs))
        self.variables += vs
    def _get_var_ordering(self,var):
        idx = self.variables.index(var)
        vs = self.variables
        # make the var first
        r = list(range(len(vs)))
        r.remove(idx)
        ordering = [idx] + r
        log.debug('ordering'+str(ordering))
        return ordering

    def get_ordered_tensors(self,var):
        log.debug(self._tensor)
        self._align_to_vars()
        return np.transpose(
            self._tensor,
            self._get_var_ordering(var)
        )
    def _align_to_vars(self):
        if len(self._tensor.shape)!=len(self.variables):
            if len(self.variables)==1:
                self._tensor = np.sum(self._tensor,axis=0)
            elif len(self.variables)==2:
                self._tensor = np.sum(
                    self._tensor,
                    # indexes are (in1,out1,in2,out2)
                    axis=(0,2)
                )

    def multiply(self,tensor,var):
        t = Tensor()
        t1 = self.get_ordered_tensors(var)
        t2 = tensor.get_ordered_tensors(var)
        print("op1 op2 and result:")
        print(t1)
        print(t2)
        _t = np.tensordot(t1,t2,axes=0)
        _t = np.sum(_t,axis=0)
        print(_t)
        t._tensor = _t
        t.add_variable(
            *(self.variables+tensor.variables)
        )
        return t
    def sum(self,axis=0):
        print(self._tensor)
        t = Tensor()
        v = self.variables[0]
        t.variables = [y for y in self.variables if y != v]
        t._tensor = np.sum(self._tensor,axis=0)
        return t
    def __repr__(self):
        return "<tensor \n "+self._tensor.__repr__()+ '\nvars: '+str(self.variables)+">"
