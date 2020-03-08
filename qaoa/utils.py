import copy
import numpy as np
import qtree
from qtree.optimizer import Var
import matplotlib.pyplot as plt

def get_neighbours_peo(old_graph):
    graph = copy.deepcopy(old_graph)
    peo = []
    nghs = []
    while graph.number_of_nodes():
        best_node, degree = sorted(graph.degree, key=lambda x:x[1])[0]
        nghs.append(degree)
        peo.append(best_node)
        qtree.graph_model.eliminate_node(graph, best_node)
    return peo, nghs

def get_locale_peo(old_graph, rule):
    # This is far below computationally effective
    graph = copy.deepcopy(old_graph)
    
    path= []
    vals = []
    while graph.number_of_nodes():
        #nodes = sorted(graph.nodes, key=int)
        nodes = list(graph.nodes)
        rule_ = lambda n: rule(graph, n)
        costs = list(map(rule_, nodes))
        _idx = np.argmin(costs)
        vals.append(costs[_idx])
        node = nodes[_idx]
        path.append(node)
        qtree.graph_model.eliminate_node(graph, node)
    return path, vals


def get_test_circ_filename(root, size, depth=10, id_=0):
    grid = f'{size}x{size}'
    return f'{root}/inst/cz_v2/{grid}/inst_{grid}_{depth}_{id_}.txt'


def test_circ(*args, **kwargs):
    test_file = get_test_circ_filename(*args, **kwargs)
    return qtree.read_circuit_file(test_file)


def reorder_graph(graph, peo):
    graph, label_dict = qtree.graph_model.relabel_graph_nodes(
        graph, dict(zip(peo, sorted(graph.nodes(), key=int)))
    )
    return graph, label_dict


def plot_cost(mems, flops):
    plt.yscale('log')
    ax = plt.gca()
    ax.grid(which='minor', alpha=0.5, linestyle='-', axis='both')
    ax.grid(which='major', alpha=0.6, axis='both')
    ax.yaxis.set_tick_params(labelbottom=True)
    ax.minorticks_on()

    plt.plot(mems, label='Memory')
    plt.plot(flops, label='FLOP')
    plt.legend()


def nodes_to_vars(old_graph, peo):
    peo_vars = [Var(v, size=old_graph.nodes[v]['size'],
                    name=old_graph.nodes[v]['name']) for v in peo]
    return peo_vars


def n_neighbors(graph, node):
    return len(list(graph.neighbors(node)))


def edges_to_clique(graph, node):
    ngh = list(graph.neighbors(node))
    N = len(ngh)
    edges = [e for e in graph.edges if all([x in ngh for x in e[:2]])]
    try:
        edges = [e for e in edges if e[2] == 0]
    except IndexError:
        pass
    return N*(N-1)//2 - len(edges)


def _neighbors(graph, node):
    return list(graph.neighbors(node))


def get_neighbours_path(old_graph):
    graph = copy.deepcopy(old_graph)
    ngh = []
    nodes = sorted(graph.nodes, key=int)
    for node in nodes:
        ngh.append(n_neighbors(graph, node))
        qtree.graph_model.eliminate_node(graph, node)
    return nodes, ngh
