import numpy as np
import re
import copy
from src.quickbb_api import gen_cnf, run_quickbb


def get_peo(graph, n_qubit_parralel=0, fix_variables=None):
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
    n_qubit_parralel : int, default 0
          parallelize over this nimber of qubits. Results in
          2**n_qubit_parralel independent jobs
    fix_variables : list, default None
          list containing edge variables which should not be
          eliminated
    Returns
    -------
    peo : list
          list containing indices in order to eliminate
    max_mem : int
          memory amount needed to perform the contraction
          (in floats)
    """

    cnffile = 'quickbb.cnf'
    gen_cnf(cnffile, graph)
    out_bytes = run_quickbb(cnffile, './quickbb/run_quickbb_64.sh')

    # Extract order
    m = re.search(b'(?P<peo>(\d+ )+).*Treewidth=(?P<treewidth>\s\d+)',
                      out_bytes, flags=re.MULTILINE | re.DOTALL )

    peo = [int(ii) for ii in m['peo'].split()]
    treewidth = int(m['treewidth'])

    return peo, 2**treewidth


def get_peo_random(old_graph, n_qubit_parralel=0, edge_variables=None):
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
    print(idx_parallel)

    return None, None
