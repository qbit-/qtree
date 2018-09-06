"""
Operations with graphical models
"""

import numpy as np
import re
import copy
import networkx as nx
import itertools

from src.quickbb_api import gen_cnf, run_quickbb
from src.logger_setup import log


def relabel_graph_nodes(graph, label_dict=None):
    """
    Relabel graph nodes to consequtive numbers. If label
    dictionary is not provided, a relabelled graph and a
    dict {new : old} will be returned. Otherwise, the graph
    is relabelled (and returned) according to the label
    dictionary and an inverted dictionary is returned.

    Parameters
    ----------
    graph : networkx.Graph
            graph to relabel
    label_dict : optional, dict-like
            dictionary for relabelling {old : new}

    Returns
    -------
    new_graph : networkx.Graph
            relabeled graph
    label_dict : dict
            {new : old} dictionary for inverse relabeling
    """
    if label_dict is None:
        label_dict = {old: num for num, old in
                      enumerate(graph.nodes(data=False), 1)}
        new_graph = nx.relabel_nodes(graph, label_dict, copy=True)
    else:
        new_graph = nx.relabel_nodes(graph, label_dict, copy=True)

    # invert the dictionary
    label_dict = {val: key for key, val in label_dict.items()}

    return new_graph, label_dict


def get_peo(old_graph):
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
          list containing indices in loptimal order of elimination
    max_mem : int
          leading memory order needed to perform the contraction (in floats)
    """

    cnffile = 'output/quickbb.cnf'
    initial_indices = old_graph.nodes()
    graph, label_dict = relabel_graph_nodes(old_graph)

    gen_cnf(cnffile, graph)
    out_bytes = run_quickbb(cnffile, './quickbb/run_quickbb_64.sh')

    # Extract order
    m = re.search(b'(?P<peo>(\d+ )+).*Treewidth=(?P<treewidth>\s\d+)',
                  out_bytes, flags=re.MULTILINE | re.DOTALL)

    peo = [int(ii) for ii in m['peo'].split()]

    # Map peo back to original indices
    peo = [label_dict[pp] for pp in peo]

    # find the rest of indices which quickBB did not spit out.
    # Those include isolated nodes (don't affect
    # scaling and may be added to the end of the variables list)
    # and something else

    isolated_nodes = nx.isolates(old_graph)
    peo = peo + sorted(isolated_nodes)

    # assert(set(initial_indices) - set(peo) == set())
    missing_indices = set(initial_indices)-set(peo)
    # The next line needs review. Why quickBB misses some indices?
    # It is here to make program work, but is it an optimal order?
    peo = peo + sorted(list(missing_indices))

    assert(sorted(peo) == sorted(initial_indices))
    log.info('Final peo from quickBB:\n{}'.format(peo))

    treewidth = int(m['treewidth'])

    return peo, 2**treewidth


def get_peo_parallel_random(old_graph, n_var_parallel=0):
    """
    Same as :py:meth:`get_peo`, but with randomly chosen nodes
    to parallelize over.

    Parameters
    ----------
    old_graph : networkx.Graph
                graph to contract (after eliminating variables which
                are parallelized over)
    n_var_parallel : int
                number of variables to eliminate by parallelization

    Returns
    -------
    peo : list
          list containing indices in loptimal order of elimination
    max_mem : int
          leading memory order needed to perform the contraction (in floats)
    idx_parallel : list
          variables removed by parallelization
    graph : networkx.Graph
          new graph without parallelized variables
    """
    graph = copy.deepcopy(old_graph)

    indices = list(graph.nodes())
    idx_parallel = np.random.choice(
        indices, size=n_var_parallel, replace=False)

    for idx in idx_parallel:
        graph.remove_node(idx)

    log.info("Removed indices by parallelization:\n{}".format(idx_parallel))

    peo, max_mem = get_peo(graph)

    return peo, max_mem, sorted(idx_parallel), graph


def get_node_by_degree(graph):
    """
    Returns a list of pairs (node : degree) for the
    provided graph. Self loops in the graph are removed (if any)
    before the calculation of degree
    """
    nodes_by_degree = list((node, degree) for
                           node, degree in nx.Graph(graph).degree())
    return nodes_by_degree


def get_node_by_betweenness(graph):
    """
    Returns a list of pairs (node : betweenness) for the
    provided graph
    """
    nodes_by_beteenness = list(
        nx.betweenness_centrality(
            nx.Graph(graph),
            normalized=False, endpoints=True).items())

    return nodes_by_beteenness


def get_peo_parallel_by_metric(
        old_graph, n_var_parallel=0, metric_fn=get_node_by_degree):
    """
    Parallel-splitted version of :py:meth:`get_peo` with nodes
    to split chosed according to the metric function. Metric
    function should take a graph and return a list of pairs
    (node : metric_value)

    Parameters
    ----------
    old_graph : networkx.Graph or networkx.MultiGraph
                graph to contract (after eliminating variables which
                are parallelized over)
    n_var_parallel : int
                number of variables to eliminate by parallelization
    metric_fn : function
                function to evaluate node metric

    Returns
    -------
    peo : list
          list containing indices in loptimal order of elimination
    max_mem : int
          leading memory order needed to perform the contraction (in floats)
    idx_parallel : list
          variables removed by parallelization
    graph : networkx.Graph or networkx.MultiGraph
          new graph without parallelized variables
    """
    graph = copy.deepcopy(old_graph)

    # get nodes by metric in descending order
    nodes_by_metric = metric_fn(graph)
    nodes_by_metric.sort(key=lambda pair: pair[1], reverse=True)

    idx_parallel = []
    for ii in range(n_var_parallel):
        node, degree = nodes_by_metric[ii]
        idx_parallel.append(node)

    for idx in idx_parallel:
        graph.remove_node(idx)

    log.info("Removed indices by parallelization:\n{}".format(idx_parallel))

    peo, max_mem = get_peo(graph)

    return peo, max_mem, sorted(idx_parallel), graph


def cost_estimator(old_graph):
    """
    Estimates the cost of the bucket elimination algorithm.
    The order of elimination is defined by node order (if ints are
    used as nodes then it will be the number of integers).

    Parameters
    ----------
    old_graph : networkx.Graph or networkx.MultiGraph
               Graph containing the information about the contraction
    Returns
    -------
    cost : list
              List of (memory, flops) pairs in order of the
              bucket elimination algorithm
    """
    graph = copy.deepcopy(old_graph)
    nodes = sorted(graph.nodes)

    cost = []
    for n, node in enumerate(nodes):
        neighbors = list(graph[node])

        memory = 2**(len(neighbors))
        flops  = 2**(len(neighbors) + 1)

        cost.append((memory, flops))

        if len(neighbors) > 1:
            edges = itertools.combinations(neighbors, 2)
        else:
            # If this is a single variable tensor, add self loop
            edges = [[neighbors[0], neighbors[0]]]

        graph.remove_node(node)
        graph.add_edges_from(edges, tensor=f'E{n}')

    return cost
