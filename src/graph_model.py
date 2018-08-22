import numpy as np
import re
import copy
import networkx as nx
from src.quickbb_api import gen_cnf, run_quickbb
from src.logger_setup import log


def relabel_graph_nodes(graph, label_dict=None):
    """
    Relabel graph nodes to consequtive numbers. If label
    dictionary is not provided, a relabelled graph and a
    dict old->new will be returned. Otherwise, the graph
    is relabelled (and returned) according to the label dictionary and
    an inverted dictionary is returned.
    """
    if label_dict is None:
        label_dict = {old : num for num, old in
                      enumerate(graph.nodes(data=False), 1)}
        new_graph = nx.relabel_nodes(graph, label_dict, copy=True)
    else:
        # invert the dictionary
        label_dict = {val : key for key, val in label_dict.items()}
        new_graph = nx.relabel_nodes(graph, label_dict, copy=True)
        
    return new_graph, label_dict


def get_peo(graph):
    """
    Calculates the elimination order for an undirected
    graphical model of the circuit. Optionally finds `n_qubit_parralel`
    qubits and splits the contraction over their values, such
    that the resulting contraction is lowest possible cost.
    Optionally fixes the values border nodes to calculate
    full state vector.

    Parameters
    ----------
    graph : networkx.Graph
          graph of the undirected graphical model to decompose
    Returns
    -------
    peo : list
          list containing indices in order to eliminate
    max_mem : int
          memory amount needed to perform the contraction
          (in floats)
    """

    cnffile = 'quickbb.cnf'
    graph, label_dict = relabel_graph_nodes(graph)
    
    gen_cnf(cnffile, graph)
    out_bytes = run_quickbb(cnffile, './quickbb/run_quickbb_64.sh')

    # Extract order
    m = re.search(b'(?P<peo>(\d+ )+).*Treewidth=(?P<treewidth>\s\d+)',
                      out_bytes, flags=re.MULTILINE | re.DOTALL )

    peo = [int(ii) for ii in m['peo'].split()]
    
    # invert the label dictionary and relabel peo back
    label_dict = {val : key for key, val in label_dict.items()}
    peo = [label_dict[pp] for pp in peo]
    
    treewidth = int(m['treewidth'])

    return peo, 2**treewidth


def get_peo_parallel_random(old_graph, n_qubit_parralel=0):
    """
    Same as above, but with randomly chosen nodes
    to parallelize. For testing only
    """
    graph = copy.deepcopy(old_graph)

    indices = list(graph.nodes())
    idx_parallel = np.random.choice(
        indices, size=n_qubit_parralel, replace=False)

    for idx in idx_parallel:
        graph.remove_node(idx)

    log.info("Removed indices by parallelization:\n{}".format(idx_parallel))

    peo, max_mem = get_peo(graph)

    # find isolated nodes as they may emerge after split
    # and are not accounted by quickbb. They don't affect
    # scaling and may be added to the end of the bucket list

    isolated_nodes = nx.isolates(graph)    
    peo = peo + sorted(isolated_nodes)
    
    return peo, max_mem, sorted(idx_parallel), graph


