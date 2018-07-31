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

def vertical_eliminate(buckets):
    pass


def circ2graph(circuit):
    g = nx.Graph()
    tensors2vars = []

    qubit_count = len(circuit[0])
    print(qubit_count)

    # we start from 0 here to avoid problems with quickbb
    for i in range(1, qubit_count+1):
        g.add_node(i)
    current_var = qubit_count
    variable_col= list(range(1,qubit_count+1))
    # Process first layer
    buckets = []
    for op in circ[0]:
        bucket = Bucket(op._qubits[0])
        bucket += op
        buckets.append(bucket)

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
                #bucket = Bucket(current_var+1)
                #bucket += op
                #buckets.append(bucket)

                variable_col[op._qubits[0]] = current_var

            if isinstance(op,cZ):
                # cZ connects two variables with an edge
                i1 = variable_col[op._qubits[0]]
                i2 = variable_col[op._qubits[1]]
                tensor2vars += i1
                tensor2vars += i2
                g.add_edge(i1,i2)

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
    return g

