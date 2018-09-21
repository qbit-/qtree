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

QUICKBB_COMMAND = './quickbb/run_quickbb_64.sh'


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
    treewidth : int
          treewidth of the decomposition
    """

    cnffile = 'output/quickbb.cnf'
    initial_indices = old_graph.nodes()
    graph, label_dict = relabel_graph_nodes(old_graph)

    if graph.number_of_edges() - graph.number_of_selfloops() > 0:
        gen_cnf(cnffile, graph)
        out_bytes = run_quickbb(cnffile, QUICKBB_COMMAND)

        # Extract order
        m = re.search(b'(?P<peo>(\d+ )+).*Treewidth=(?P<treewidth>\s\d+)',
                      out_bytes, flags=re.MULTILINE | re.DOTALL)

        peo = [int(ii) for ii in m['peo'].split()]

        # Map peo back to original indices
        peo = [label_dict[pp] for pp in peo]

        treewidth = int(m['treewidth'])
    else:
        peo = []
        treewidth = 1

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

    return peo, treewidth


def split_graph_random(old_graph, n_var_parallel=0):
    """
    Splits a graphical model with randomly chosen nodes
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

    peo, treewidth = get_peo(graph)

    return sorted(idx_parallel), graph


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
    memory : list
              Memory cost for steps of the bucket elimination algorithm
    flops : list
              Flop cost for steps of the bucket elimination algorithm
    """
    graph = copy.deepcopy(old_graph)
    nodes = sorted(graph.nodes)

    results = []
    for n, node in enumerate(nodes):
        neighbors = list(graph[node])

        # We have to find all unique tensors which will be contracted
        # in this bucket. They label the edges coming from
        # the current node (may be multiple edges between
        # the node and its neighbor).
        # Then we have to count only the number of unique tensors.
        if graph.is_multigraph():
            edges_from_node = [list(graph[node][neighbor].values())
                               for neighbor in neighbors]
            tensor_hash_tags = [edge['hash_tag'] for edges_of_neighbor
                                in edges_from_node
                                for edge in edges_of_neighbor]
        else:
            tensor_hash_tags = [graph[node][neighbor]['hash_tag']
                                for neighbor in neighbors]

        n_unique_tensors = len(set(tensor_hash_tags))
        if n_unique_tensors == 0:
            n_multiplications = 0
        else:
            n_multiplications = n_unique_tensors - 1

        memory = 2**(len(neighbors))

        # there are number_of_terms - 1 multiplications and 1 addition
        # repeated size_of_the_result*size_of_contracted_variables
        # times for each contraction
        flops = (2**(len(neighbors) + 1)       # this is addition
                 + 2**(len(neighbors) + 1)*n_multiplications)

        results.append((memory, flops))

        if len(neighbors) > 1:
            edges = itertools.combinations(neighbors, 2)
        elif len(neighbors) == 1:
            # This node had a single neighbor, add self loop to it
            edges = [[neighbors[0], neighbors[0]]]
        else:
            # This node had no neighbors
            edges = None

        graph.remove_node(node)

        if edges is not None:
            graph.add_edges_from(
                edges, tensor=f'E{n}',
                hash_tag=hash((f'E{n}', tuple(neighbors))))

    return tuple(zip(*results))


def get_node_by_degree(graph):
    """
    Returns a list of pairs (node : degree) for the
    provided graph.

    Parameters
    ----------
    graph : networkx.Graph without self-loops and parallel edges

    Returns
    -------
    nodes_by_degree : dict
    """
    nodes_by_degree = list((node, degree) for
                           node, degree in graph.degree())
    return nodes_by_degree


def get_node_by_betweenness(graph):
    """
    Returns a list of pairs (node : betweenness) for the
    provided graph

    Parameters
    ----------
    graph : networkx.Graph without self-loops and parallel edges

    Returns
    -------
    nodes_by_beteenness : dict
    """
    nodes_by_beteenness = list(
        nx.betweenness_centrality(
            graph,
            normalized=False, endpoints=True).items())

    return nodes_by_beteenness


def get_node_by_mem_reduction(old_graph):
    """
    Returns a list of pairs (node : reduction_in_flop_cost) for the
    provided graph. This is the algorithm Alibaba used

    Parameters
    ----------
    graph : networkx.Graph without self-loops and parallel edges

    Returns
    -------
    nodes_by_degree : dict
    """
    # First find the initial flop cost
    # Find elimination order
    peo, treewidth = get_peo(old_graph)
    number_of_nodes = len(peo)

    # Transform graph to this order
    graph, label_dict = relabel_graph_nodes(
        old_graph, dict(zip(range(1, number_of_nodes+1), peo))
    )

    # Get flop cost of the bucket elimination
    initial_mem, initial_flop = cost_estimator(graph)

    nodes_by_flop_reduction = []
    for node in graph.nodes(data=False):
        reduced_graph = copy.deepcopy(graph)
        # Take out one node
        reduced_graph.remove_node(node)
        # Renumerate graph nodes to be consequtive ints (may be redundant)
        order = (list(range(1, node))
                 + list(range(node + 1, number_of_nodes + 1)))
        reduced_graph, _ = relabel_graph_nodes(
            reduced_graph, dict(zip(range(1, number_of_nodes), order))
        )
        mem, flop = cost_estimator(reduced_graph)
        delta = np.sum(initial_mem) - np.sum(mem)

        # Get original node number for this node
        old_node = label_dict[node]

        nodes_by_flop_reduction.append((old_node, delta))

    return nodes_by_flop_reduction


def split_graph_by_metric(
        old_graph, n_var_parallel=0, metric_fn=get_node_by_degree):
    """
    Parallel-splitted version of :py:meth:`get_peo` with nodes
    to split chosen according to the metric function. Metric
    function should take a graph and return a list of pairs
    (node : metric_value)

    Parameters
    ----------
    old_graph : networkx.Graph or networkx.MultiGraph
                graph to split by parallelizing over variables
                and to contract

                Parallel edges and self-loops in the graph are
                removed (if any) before the calculation of metric

    n_var_parallel : int
                number of variables to eliminate by parallelization
    metric_fn : function
                function to evaluate node metric

    Returns
    -------
    idx_parallel : list
          variables removed by parallelization
    graph : networkx.Graph or networkx.MultiGraph
          new graph without parallelized variables
    """
    graph = nx.Graph(copy.deepcopy(old_graph))
    graph.remove_edges_from(graph.selfloop_edges())

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

    return sorted(idx_parallel), graph


def split_graph_with_mem_constraint(
        old_graph,
        mem_constraint,
        metric_function=get_node_by_degree,
        step_by=5):
    """
    Calculates memory cost vs the number of parallelized
    variables for a given graph.

    Parameters
    ----------
    old_graph : networkx.Graph()
           initial contraction graph
    mem_constraint : int
           Upper limit on memory
    metric_function : function, optional
           function to rank nodes for elimination
    step_by : int, optional
           scan the metric function with this step
    Returns
    -------
    n_var_parallel : int
           number of the variables to parallelize
    """

    n_var_total = old_graph.number_of_nodes()

    for n_var_parallel in range(0, n_var_total, step_by):
        idx_parallel, reduced_graph = split_graph_by_metric(
             old_graph, n_var_parallel, metric_fn=metric_function)

        peo, treewidth = get_peo(reduced_graph)

        graph_parallel, label_dict = relabel_graph_nodes(
            reduced_graph, dict(zip(range(1, len(peo) + 1), peo))
        )

        mem_cost, flop_cost = cost_estimator(graph_parallel)

        max_mem = sum(mem_cost)

        if max_mem < mem_constraint:
            break

    return idx_parallel, reduced_graph
