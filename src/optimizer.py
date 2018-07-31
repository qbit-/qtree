import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import logging as log
from .operators import *

class Bucket():
    tensors = []
    def __init__(self, var):
        self.var = var
    def set_var(self,var):
        self.var = var
    def append(self,t):
        self.tensors.append(t)
    def __iadd__(self,t):
        self.append(t)
    def process(self):
        result = self.tensors[0].matrix
        for op in self.tensors:
            result = np.multiply(result,op.matrix,
                                axes=[(-1,),(index),()]
                                )

        return np.sum(result, axis = )

class Tensor2Vars():
    def __init__(self,op):
        self.op = op
        self.tensor = op.tensor
        self.vars = []
    def append_var(self,var):
        self.vars.append(var)
    def __iadd__(self,var):
        self.append_var(var)
class Variable2Qubitidx():
    storage = {}
    def __init__(self):
        pass
    def add_variable_idx(self,variable,idx):
        try:
            self.storage[idx].append(variable)
        except:
            self.storage[idx] = [variable]
    def get_qubit_idx_of_var(self,var):
        for index,variables self.storage.items():
            if variable in variables:
                return index


def find_tensors_by_var(tensors2vars,var):
    tensors = []
    for t in tensors2vars:
        if var in t.vars
        tensors.append(t)
    return tensors

def get_transpose_order(t,qidx):
    if isinstance( t.op, cZ):
        idx = t.op._qubits
        # Not sure if it's actually works
        if idx[0]== qidx:
            return (0,1,2,3)
        else:
            return (1,3,0,2)
    else:
        if idx[0]== qidx:
            return (0,1)
        else:
            return (1,0)

def naive_eliminate(graph,tensors2vars,v2qidx):
    for variable in graph.get_nodes():
        tensors = find_tensors_by_var(variable)
        product = tensors[0]
        for t in tensors[1:]:
            qidx = v2qidx.get_qubit_idx_of_var(
                        variable
                    )

            t_ordered = np.transpose(
                t.tensor,
                get_transpose_order(t,qidx)
            )
            product = np.multiply(product,t_ordered)
        new_tensor = np.sum(product,axis=0)
        ten


def circ2graph(circuit):
    g = nx.Graph()
    tensors2vars = []

    qubit_count = len(circuit[0])
    print(qubit_count)

    # we start from 0 here to avoid problems with quickbb
    tensors2vars = []
    v2qidx = Variable2Qubitidx()

    # Process first layer
    for i in range(1, qubit_count+1):
        g.add_node(i)
        tv = Tensor2Vars(op)
        tv += i
        tensors2vars.append(tv)
        v2qidx.add_variable_idx(i,i)

    current_var = qubit_count
    variable_col= list(range(1,qubit_count+1))


    variable_col= list(range(1, qubit_count+1))
    for layer in circuit[1:-1]:
        for op in layer:
            tensor2vars = Tensor2Vars(op)
            if not op.diagonal:
                # Non-diagonal gate adds a new variable and
                # an edge to graph
                g.add_node(current_var+1)
                g.add_edge(
                    variable_col[op._qubits[0]],
                    current_var+1 )
                current_var += 1
                tensor2vars += current_var
                tensor2vars += variable_col[op._qubits[0]]
                v2qidx.add_variable_idx(
                    variable_col[op._qubits[0]],
                    current_var
                )

                variable_col[op._qubits[0]] = current_var

            elif isinstance(op,cZ):
                # cZ connects two variables with an edge
                i1 = variable_col[op._qubits[0]]
                i2 = variable_col[op._qubits[1]]
                tensor2vars += i1
                tensor2vars += i2
                g.add_edge(i1,i2)
            # tensors2 vars is a list of tensos
            # which leads to variables it operates on
            elif:
                tensor2vars += current_var
            tensors2vars.append(tensor2vars)
    v = g.number_of_nodes()
    e = g.number_of_edges()
    print(g)
    log.info(f"Generated graph with {v} nodes and {e} edges")
    log.info(f"last index contains from {variable_col}")

    aj = nx.adjacency_matrix(g)
    matfile = 'adjacency_graph.mat'
    np.savetxt(matfile ,aj.toarray(),delimiter=" ",fmt='%i')
    with open(matfile,'r') as fp:
        s = fp.read()
        #s = s.replace(' ','-')
        print(s.replace('0','-'))

    plt.figure(figsize=(10,10))
    nx.draw(g,with_labels=True)
    plt.savefig('graph.eps')
    return g,tensors2vars,v2qidx

