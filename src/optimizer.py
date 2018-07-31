import networkx as nx
#import matplotlib.pyplot as plt
import numpy as np
from .Tensor import Tensor
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



def find_tensors_by_var(tensors,var):
    ts = []
    for t in tensors:
        if var in t.variables:
         ts.append(t)
    return ts

def naive_eliminate(graph,tensors):
    for variable in graph.nodes():
        log.info(f"eliminating {variable} var")
        _tensors = find_tensors_by_var(tensors,variable)
        l = len(_tensors)
        v = str(([t.variables for t in _tensors]))
        log.debug(f'need to multiply {l} tensors {v}')
        product = _tensors[0]
        tensors.remove(_tensors[0])
        for t in _tensors[1:]:
            product = product.multiply(t,variable)
            tensors.remove(t)
        new_tensor = product.sum(over=variable)
        tensors.append(new_tensor)
        log.debug('new tensor'+str(new_tensor))


def circ2graph(circuit):
    g = nx.Graph()
    tensors2vars = []

    qubit_count = len(circuit[0])
    print(qubit_count)

    # we start from 0 here to avoid problems with quickbb
    tensors =[]

    # Process first layer
    for i in range(1, qubit_count+1):
        g.add_node(i)
        tensor =Tensor(circuit[0][i-1])
        # 0 is inital index
        tensor.add_variable(0,i)

    current_var = qubit_count
    variable_col= list(range(1,qubit_count+1))

    for layer in circuit[1:-1]:
        for op in layer:
            tensor = Tensor(op)
            if not op.diagonal:
                # Non-diagonal gate adds a new variable and
                # an edge to graph
                g.add_node(current_var+1)
                g.add_edge(
                    variable_col[op._qubits[0]],
                    current_var+1 )
                tensor.add_variable(
                    variable_col[op._qubits[0]],
                    current_var+1 )
                current_var += 1

                variable_col[op._qubits[0]] = current_var

            elif isinstance(op,cZ):
                # cZ connects two variables with an edge
                i1 = variable_col[op._qubits[0]]
                i2 = variable_col[op._qubits[1]]
                g.add_edge(i1,i2)
                tensor.add_variable(i1,i2)
            # tensors2 vars is a list of tensos
            # which leads to variables it operates on
            else:
                tensor.add_variable( current_var)
            tensors.append(tensor)
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
        #print(s.replace('0','-'))

    #plt.figure(figsize=(10,10))
    nx.draw(g,with_labels=True)
    #plt.savefig('graph.eps')
    return g,tensors

