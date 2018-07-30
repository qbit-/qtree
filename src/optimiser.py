
#from igraph import *
from graph_tool.all import *
import graph_tool
import numpy as np
import logging as log
from .operators import *

def circ2graph(circ):
    vertices = []
    edges = []
    g = Graph(directed = False)

    qubit_count = len(circ[0])
    vertices = range(qubit_count)
    print(qubit_count)
    current_var = qubit_count-1
    variable_col= list(range(qubit_count))

    for i in range(qubit_count):
        g.add_vertex()

    for layer in circ[1:-1]:
        print(layer)
        for op in layer:
            if not op.diagonal:
                # Non-diagonal gate adds a vertex to graph
                #vertices.append(current_var+1)
                #edges.append( (current_var,current_var+1) )
                #g.add_vertices(current_var+1)
                #g.add_edges( (current_var,current_var+1) )
                v = g.add_vertex()
                g.add_edge( g.vertex(variable_col[op.qubit_idx]),v)
                current_var += 1

                variable_col[op.qubit_idx] = current_var

            if isinstance(op,cZ):
                # Hadamard connects two variables with edge
                #edges.append(
                #g.add_edges(
                g.add_edge(
                    g.vertex(variable_col[op.qubit_idx[0]]),
                    g.vertex(variable_col[op.qubit_idx[1]])
                )
    v = len(list(g.vertices()))
    e = len(list(g.edges()))
    print(g)
    log.info(f"Generated graph with {v} nodes and {e} edges")
    #pos = graph_tool.draw.sfdp_layout(g)
    #print(graph_tool.draw)
    #graph_tool.draw.cairo_draw(g, pos=pos, output="graph-draw-sfdp.pdf")
    aj = graph_tool.spectral.adjacency(g)
    matfile = 'adjacency_graph.mat'
    np.savetxt(matfile ,aj.toarray(),delimiter=" ",fmt='%i')
    with open(matfile,'r') as fp:
        s = fp.read()
        #s = s.replace(' ','-')
        print(s.replace('0','-'))
    return g

