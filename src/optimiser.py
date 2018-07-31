
import networkx as nx
#import graph_tool
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


def vertical_eliminate(buckets):
    pass


def circ2graph(circ):
    vertices = []
    edges = []
    g = nx.Graph()

    qubit_count = len(circ[0])
    vertices = range(qubit_count)
    print(qubit_count)

    for i in range(1,qubit_count):
        g.add_node(i)
    current_var = qubit_count
    variable_col= list(range(1,qubit_count+1))
    bucket = Bucket(variable_col)
    # Process first layer
    for op in circ[0]:
        bucket += op
    buckets = [bucket]

    for layer in circ[1:-1]:
        print(layer)
        bucket = Bucket()
        for op in layer:
            if not op.diagonal:
                # Non-diagonal gate adds a vertex to graph
                #vertices.append(current_var+1)
                #edges.append( (current_var,current_var+1) )
                g.add_node(current_var+1 )
                g.add_edge(
                    variable_col[op._qubits[0]],
                    current_var+1 )
                current_var += 1

                bucket +=op

                variable_col[op._qubits[0]] = current_var

            if isinstance(op,cZ):
                # Hadamard connects two variables with edge
                #edges.append(
                #g.add_edges(
                g.add_edge(
                    variable_col[op._qubits[0]],
                    variable_col[op._qubits[1]]
                )
        bucket.set_var(variable_col)
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

