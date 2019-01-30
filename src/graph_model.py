"""
Operations with graphical models
"""

import numpy as np
import re
import copy
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import random
import os

from collections import Counter

import src.system_defs as defs
import src.utils as utils
from src.quickbb_api import gen_cnf, run_quickbb
from src.logger_setup import log

random.seed(0)


def read_graph(filename, max_depth=None):
    """
    Reads circuit from filename and builds its contraction graph

    Parameters
    ----------
    filename : str
             circuit file in the format of Sergio Boixo
    max_depth : int
             maximal depth of gates to read

    Returns
    -------
    qubit_count : int
            number of qubits in the circuit
    graph : networkx.MultiGraph
    """
    graph = nx.MultiGraph()

    # perform the cirquit file processing
    log.info(f'reading file {filename}')

    with open(filename, 'r') as fp:
        # read the number of qubits
        qubit_count = int(fp.readline())
        log.info("There are {:d} qubits in circuit".format(qubit_count))

        n_ignored_layers = 0
        current_layer = 0

        # initialize the variables and add nodes to graph
        for i in range(1, qubit_count+1):
            graph.add_node(i, name=utils.num_to_alnum(i))
        layer_variables = list(range(1, qubit_count+1))
        current_var = qubit_count

        # Add selfloops to input nodes
        for ii in range(1, qubit_count+1):
            graph.add_edge(
                ii, ii,
                tensor=f'I{ii}',
                hash_tag=hash((f'I{ii}', (ii, ii), random.random())))

        for idx, line in enumerate(fp):

            # Read circuit layer by layer. Decipher contents of the line
            m = re.search(r'(?P<layer>[0-9]+) (?P<operation>h|t|cz|x_1_2|y_1_2) (?P<qubit1>[0-9]+) ?(?P<qubit2>[0-9]+)?', line)
            if m is None:
                raise Exception("file format error at line {}".format(idx))
            layer_num = int(m.group('layer'))

            # Skip layers if max_depth is set
            if max_depth is not None and layer_num > max_depth:
                n_ignored_layers = layer_num - max_depth
                continue
            if layer_num > current_layer:
                current_layer = layer_num

            op_identif = m.group('operation')
            if m.group('qubit2') is not None:
                q_idx = int(m.group('qubit1')), int(m.group('qubit2'))
            else:
                q_idx = (int(m.group('qubit1')),)

            # Now apply what we got to build the graph
            if op_identif == 'cz':
                # cZ connects two variables with an edge
                var1 = layer_variables[q_idx[0]]
                var2 = layer_variables[q_idx[1]]
                graph.add_edge(var1, var2,
                               tensor=op_identif,
                               hash_tag=hash((op_identif, (var1, var2),
                                             random.random())))

            # Skip Hadamard tensors - for now
            elif op_identif == 'h':
                pass

            # Add selfloops for single variable gates
            elif op_identif == 't':
                var1 = layer_variables[q_idx[0]]
                graph.add_edge(var1, var1,
                               tensor=op_identif,
                               hash_tag=hash((op_identif, (var1, var1),
                                             random.random())))
            # Process non-diagonal gates X and Y
            else:
                var1 = layer_variables[q_idx[0]]
                var2 = current_var+1
                graph.add_node(var2, name=utils.num_to_alnum(var2))
                graph.add_edge(var1, var2,
                               tensor=op_identif,
                               hash_tag=hash((op_identif, (var1, var2),
                                             random.random())))
                current_var += 1
                layer_variables[q_idx[0]] = current_var

        # Add selfloops to output nodes
        for q_idx, var in zip(range(1, qubit_count+1), layer_variables):
            graph.add_edge(
                var, var,
                tensor=f'O{q_idx}',
                hash_tag=hash((f'O{q_idx}', (var, var), random.random())))

        # We are done, print stats
        if n_ignored_layers > 0:
            log.info("Ignored {} layers".format(n_ignored_layers))

        v = graph.number_of_nodes()
        e = graph.number_of_edges()

        log.info(f"Generated graph with {v} nodes and {e} edges")
        log.info(f"last index contains from {layer_variables}")

    return qubit_count, graph


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


def get_simple_graph(old_graph):
    """
    Simplifies graph: MultiGraphs are converted to Graphs,
    selfloops are removed
    """
    graph = nx.Graph(old_graph, copy=True)
    graph.remove_edges_from(graph.selfloop_edges())

    return graph


def get_peo(old_graph,
            quickbb_extra_args=" --time 60 --min-fill-ordering ",
            input_suffix=None, keep_input=False):
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
    quickbb_extra_args : str, default '--min-fill-ordering --time 60'
             Optional commands to QuickBB.
    input_suffix : str, default None
             Optional suffix for the folders. If None is provided a random
             suffix is generated
    keep_input : bool, default False
             Whether to keep input files for debugging
    Returns
    -------
    peo : list
          list containing indices in loptimal order of elimination
    treewidth : int
          treewidth of the decomposition
    """

    if input_suffix is None:
        input_suffix = ''.join(str(random.randint(0, 9))
                               for n in range(8))
    cnffile = 'output/quickbb.' + input_suffix + '.cnf'
    initial_indices = old_graph.nodes()
    graph, label_dict = relabel_graph_nodes(old_graph)

    if graph.number_of_edges() - graph.number_of_selfloops() > 0:
        gen_cnf(cnffile, graph)
        out_bytes = run_quickbb(cnffile, defs.QUICKBB_COMMAND)

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

    # remove input file to honor EPA
    if not keep_input:
        os.remove(cnffile)

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
    log.info("Removed {} variables".format(len(idx_parallel)))
    peo, treewidth = get_peo(graph)

    return sorted(idx_parallel), graph


def get_cost_by_node(graph, node):
    """
    Outputs the cost corresponding to the
    contraction of the node in the graph

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
               Graph containing the information about the contraction
    node : node of the graph (such that graph can be indexed by it)

    Returns
    -------
    memory : int
              Memory cost for contraction of node
    flops : int
              Flop cost for contraction of node
    """
    neighbors = list(graph[node])
    neighbors_wo_selfloops = copy.copy(neighbors)

    # Delete node itself from the list of its neighbors.
    # This eliminates possible self loop
    while node in neighbors_wo_selfloops:
        neighbors_wo_selfloops.remove(node)

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

    # Now find all self loops (single variable tensors)
    if graph.is_multigraph():
        selfloops_from_node = [list(graph[node][neighbor].values())
                               for neighbor in neighbors
                               if neighbor == node]
        selfloop_tensor_hash_tags = [edge['hash_tag']
                                     for selfloop_of_node
                                     in selfloops_from_node
                                     for edge in selfloop_of_node]
    else:
        selfloop_tensor_hash_tags = [graph[node][neighbor]['hash_tag']
                                     for neighbor in neighbors
                                     if neighbor == node]

    # The order of tensor in each term is the number of neighbors
    # having edges with the same hash tag + 1 (the node itself),
    # except for self-loops, where the order is 1
    neighbor_tensor_orders = {}
    for hash_tag, count in Counter(tensor_hash_tags).items():
        if hash_tag in selfloop_tensor_hash_tags:
            tensor_order = {hash_tag: count}
        else:
            tensor_order = {hash_tag: count+1}
        neighbor_tensor_orders.update(tensor_order)

    # memory estimation: the size of the result + all sizes of terms
    memory = 2**(len(neighbors_wo_selfloops))
    for order in neighbor_tensor_orders.values():
        memory += 2**order

    n_unique_tensors = len(set(tensor_hash_tags))

    # there are number_of_terms - 1 multiplications
    if n_unique_tensors == 0:
        n_multiplications = 0
    else:
        n_multiplications = n_unique_tensors - 1

    size_of_the_result = len(neighbors_wo_selfloops)

    # There are n_multiplications and 1 addition
    # repeated size_of_the_result*size_of_contracted_variables
    # times for each contraction
    flops = (2**(size_of_the_result + 1)       # this is addition
             + 2**(size_of_the_result + 1)*n_multiplications)

    return memory, flops


def eliminate_node(graph, node, self_loops=True):
    """
    Eliminates node according to the tensor contraction rules.
    A new clique is formed, which includes all neighbors of the node.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
            Graph containing the information about the contraction
            GETS MODIFIED IN THIS FUNCTION
    node : node to contract (such that graph can be indexed by it)
    self_loops : bool
           Whether to create selfloops on the neighbors. Default True.

    Returns
    -------
    None
    """

    neighbors = list(graph[node])

    # Delete node itself from the list of its neighbors.
    # This eliminates possible self loop
    while node in neighbors:
        neighbors.remove(node)

    if len(neighbors) > 1:
        edges = itertools.combinations(neighbors, 2)
    elif len(neighbors) == 1 and self_loops:
        # This node had a single neighbor, add self loop to it
        edges = [[neighbors[0], neighbors[0]]]
    else:
        # This node had no neighbors
        edges = None

    graph.remove_node(node)

    if edges is not None:
        graph.add_edges_from(
            edges, tensor=f'E{node}',
            hash_tag=hash((f'E{node}', tuple(neighbors), random.random())))


def get_mem_requirement(graph):
    """
    Calculates memory to store the tensor network
    expressed in the graph model form.

    Parameters
    ----------
    graph : networkx.MultiGraph
             Graph of the network
    Returns
    -------
    memory : int
            Amount of memory
    """
    nodes = list(graph.nodes)
    nodes_wo_selfloops = copy.copy(nodes)

    # Delete node itself from the list of its neighbors.
    # This eliminates possible self loop
    while nodes in nodes_wo_selfloops:
        nodes_wo_selfloops.remove(nodes)

    # We have to find all unique tensors in the network
    # They label the edges of the graph
    # (may be multiple edges between
    # the node and its neighbor).
    # Then we have to count only the number of unique tensors.
    tensor_hash_tags = []
    selfloop_tensor_hash_tags = []
    for edge in graph.edges:
        tensor_hash_tags.append(graph.edges[edge]['hash_tag'])
        if edge[0] == edge[1]:
            selfloop_tensor_hash_tags.append(
                graph.edges[edge]['hash_tag'])

    # The order of tensor is the number of same hash tags
    tensor_orders = {}
    for hash_tag, count in Counter(tensor_hash_tags).items():
        tensor_order = {hash_tag: count}
        tensor_orders.update(tensor_order)

    # memory estimation
    memory = 0
    for order in tensor_orders.values():
        memory += 2**order

    return memory


def cost_estimator(old_graph, free_vars=[]):
    """
    Estimates the cost of the bucket elimination algorithm.
    The order of elimination is defined by node order (if ints are
    used as nodes then it will be the values of integers).

    Parameters
    ----------
    old_graph : networkx.Graph or networkx.MultiGraph
               Graph containing the information about the contraction
    free_vars : list, optional
               Nodes that will be skipped
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
        if node not in free_vars:
            memory, flops = get_cost_by_node(graph, node)
            results.append((memory, flops))
            eliminate_node(graph, node)

    # Estimate cost of the last tensor product if subsets of
    # amplitudes were evaluated
    if len(free_vars) > 0:
        size_of_the_result = len(free_vars)
        tensor_orders = [
            subgraph.number_of_nodes()
            for subgraph
            in nx.components.connected_component_subgraphs(graph)]
        # memory estimation: the size of the result + all sizes of terms
        memory = 2**size_of_the_result
        for order in tensor_orders:
            memory += 2**order
        # there are number of tensors - 1 multiplications
        n_multiplications = len(tensor_orders) - 1

        # There are n_multiplications repeated size of the result
        # times
        flops = 2**size_of_the_result*n_multiplications

        results.append((memory, flops))

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
    nodes_by_betweenness = list(
        nx.betweenness_centrality(
            graph,
            normalized=False, endpoints=True).items())

    return nodes_by_betweenness


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
        old_graph, dict(zip(peo, range(1, number_of_nodes+1)))
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
            reduced_graph, dict(zip(order, range(1, number_of_nodes)))
        )
        mem, flop = cost_estimator(reduced_graph)
        delta = np.sum(initial_mem) - np.sum(mem)

        # Get original node number for this node
        old_node = label_dict[node]

        nodes_by_flop_reduction.append((old_node, delta))

    return nodes_by_flop_reduction


def split_graph_by_metric(
        old_graph, n_var_parallel=0,
        metric_fn=get_node_by_degree,
        forbidden_nodes=[]):
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
    metric_fn : function, optional
                function to evaluate node metric.
                Default get_node_by_degree
    forbidden_nodes : list, optional
                nodes in this list will not be considered
                for deletion. Default [].
    Returns
    -------
    idx_parallel : list
          variables removed by parallelization
    graph : networkx.Graph
          new graph without parallelized variables
    """
    graph = get_simple_graph(old_graph)

    # get nodes by metric in descending order
    nodes_by_metric = metric_fn(graph)
    nodes_by_metric.sort(key=lambda pair: pair[1], reverse=True)

    nodes_by_metric_allowed = []
    for node, metric in nodes_by_metric:
        if node not in forbidden_nodes:
            nodes_by_metric_allowed.append((node, metric))

    idx_parallel = []
    for ii in range(n_var_parallel):
        node, metric = nodes_by_metric_allowed[ii]
        idx_parallel.append(node)

    for idx in idx_parallel:
        graph.remove_node(idx)

    log.info("Removed indices by parallelization:\n{}".format(idx_parallel))
    log.info("Removed {} variables".format(len(idx_parallel)))

    return sorted(idx_parallel), graph


def split_graph_with_mem_constraint(
        old_graph,
        n_var_parallel_min=0,
        mem_constraint=defs.MAXIMAL_MEMORY,
        metric_function=get_node_by_degree,
        step_by=5,
        n_var_parallel_max=None):
    """
    Calculates memory cost vs the number of parallelized
    variables for a given graph.

    Parameters
    ----------
    old_graph : networkx.Graph()
           initial contraction graph
    n_var_parallel_min : int
           minimal number of variables to split the task to
    mem_constraint : int
           Upper limit on memory per task
    metric_function : function, optional
           function to rank nodes for elimination
    step_by : int, optional
           scan the metric function with this step
    n_var_parallel_max : int, optional
           constraint on the maximal number of parallelized
           variables. Default None
    Returns
    -------
    idx_parallel : list
             list of removed variables
    reduced_graph : networkx.Graph
             reduced contraction graph
    """

    n_var_total = old_graph.number_of_nodes()
    if n_var_parallel_max is None:
        n_var_parallel_max = n_var_total

    mem_cost, flop_cost = cost_estimator(copy.deepcopy(old_graph))
    max_mem = sum(mem_cost)

    for n_var_parallel in range(n_var_parallel_min,
                                n_var_parallel_max, step_by):
        idx_parallel, reduced_graph = split_graph_by_metric(
             old_graph, n_var_parallel, metric_fn=metric_function)

        peo, treewidth = get_peo(reduced_graph)

        graph_parallel, label_dict = relabel_graph_nodes(
            reduced_graph, dict(zip(peo, range(1, len(peo) + 1)))
        )

        mem_cost, flop_cost = cost_estimator(graph_parallel)

        max_mem = sum(mem_cost)

        if max_mem <= mem_constraint:
            break

    if max_mem > mem_constraint:
        raise ValueError('Maximal memory constraint is not met')

    return idx_parallel, reduced_graph


def split_graph_dynamic_greedy(
        old_graph, n_var_parallel=0, metric_fn=get_node_by_mem_reduction,
        greedy_step_by=1):
    """
    This function splits graph by greedily selecting next nodes
    using the metric function and recomputing PEO after
    each node elimination
    """
    # Simplify graph
    graph = get_simple_graph(old_graph)

    idx_parallel = []
    for ii in range(0, n_var_parallel, greedy_step_by):
        # Get optimal order
        peo, tw = get_peo(graph)
        graph_optimal, inverse_order = relabel_graph_nodes(
            graph, dict(zip(peo, range(1, len(peo)+1))))

        # get nodes by metric in descending order
        nodes_by_metric_optimal = metric_fn(graph_optimal)
        nodes_by_metric_optimal.sort(
            key=lambda pair: pair[1], reverse=True)

        # Take first nodes by cost and map them back to original
        # order
        nodes_optimal, costs = zip(
            *nodes_by_metric_optimal[:greedy_step_by])
        nodes = [inverse_order[n] for n in nodes_optimal]

        # Update list and delete nodes
        idx_parallel += nodes
        graph.remove_nodes_from(nodes)

    return idx_parallel, graph


def draw_graph(graph, filename):
    """
    Draws graph with spectral layout
    Parameters
    ----------
    graph : networkx.Graph
            graph to draw
    filename : str
            filename for image output
    """
    plt.figure(figsize=(10, 10))
    # pos = nx.spectral_layout(graph)
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos,
            node_color=(list(graph.nodes())),
            node_size=100,
            cmap=plt.cm.Blues,
            with_labels=True,
    )
    plt.savefig(filename)


def wrap_general_graph_for_qtree(graph):
    """
    Modifies a general networkx graph to be compatible with
    graph functions from qtree. Basically, we just renumerate nodes
    from 1 and set attributes.

    Parameters
    ----------
    graph : networkx.Graph or networkx.Multigraph
            Input graph
    Returns
    -------
    new_graph : type(graph)
            Modified graph
    """
    # relabel nodes starting from 1
    label_dict = dict(zip(
        list(sorted(graph.nodes)),
        range(1, graph.number_of_nodes()+1)
    ))

    # Add unique hash tags to edges
    new_graph = nx.relabel_nodes(graph, label_dict, copy=True)
    for edge in new_graph.edges():
        new_graph.edges[edge].update({'hash_tag': hash(random.random())})
    return new_graph


def generate_random_graph(n_nodes, n_edges):
    """
    Generates a random graph with n_nodes and n_edges. Edges are
    selected randomly from a uniform distribution over n*(n-1)/2
    possible edges

    Parameters
    ----------
    n_nodes : int
          Number of nodes
    n_edges : int
          Number of edges
    Returns
    -------
    graph : networkx.Graph
          Random graph usable by graph_models
    """

    # Create a disconnected graph
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))

    # Add edges
    row, col = np.tril_indices(n_nodes)
    idx_to_pair = dict(zip(
        range(int(n_nodes*(n_nodes+1)//2)),
        zip(row, col)
    ))

    edge_indices = np.random.choice(
        range(int(n_nodes*(n_nodes+1)//2)),
        n_edges,
        replace=False
    )
    graph.add_edges_from(idx_to_pair[idx] for idx in edge_indices)

    return wrap_general_graph_for_qtree(graph)


def get_treewidth_from_peo(old_graph, peo):
    """
    This function checks the treewidth of a given peo.
    The graph is simplified: all selfloops and parallel
    edges are removed.

    Parameters
    ----------
    old_graph : networkx.Graph or networkx.MultiGraph
            graph to use
    peo : list
            list of nodes in the perfect elimination order

    Returns
    -------
    treewidth : int
            treewidth corresponding to peo
    """
    # Copy graph and make it simple
    graph = get_simple_graph(old_graph)

    treewidth = 0
    for node in peo:
        # Get the size of the next clique - 1
        neighbors = list(graph[node])
        n_neighbors = len(neighbors)
        if len(neighbors) > 1:
            edges = itertools.combinations(neighbors, 2)
        else:
            edges = None

        # Treewidth is the size of the maximal clique - 1
        treewidth = max(n_neighbors, treewidth)

        graph.remove_node(node)

        # Make the next clique
        if edges is not None:
            graph.add_edges_from(
                edges, tensor=f'E{node}',
                hash_tag=hash((f'E{node}',
                               tuple(neighbors),
                               random.random())))
    return treewidth


def make_clique_on(old_graph, clique_nodes, name_prefix='C'):
    """
    Adds a clique on the specified indices. No checks is
    done whether some edges exist in the clique. The name
    of the clique is formed from the name_prefix and the
    lowest element in the clique_nodes

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
            graph to modify
    clique_nodes : list
            list of nodes to include into clique
    name_prefix : str
            prefix for the clique name
    Returns
    -------
    new_graph : type(graph)
            New graph with clique
    """
    graph = copy.deepcopy(old_graph)

    if len(clique_nodes) == 0:
        return graph

    edges = [tuple(sorted(edge)) for edge in
             itertools.combinations(clique_nodes, 2)]
    node = min(clique_nodes)
    graph.add_edges_from(edges, tensor=name_prefix + f'{node}',
                         hash_tag=hash((name_prefix + f'{node}',
                                        tuple(clique_nodes),
                                        random.random())))
    return graph


def get_fillin_graph(old_graph, peo):
    """
    Provided a graph and an order of its indices, returns a
    triangulation of that graph corresponding to the order.

    Parameters
    ----------
    old_graph : nx.Graph or nx.MultiGraph
                graph to triangulate
    peo : elimination order to use for triangulation

    Returns
    -------
    nx.Graph or nx.MultiGraph
                triangulated graph
    """
    # get a copy of graph in the elimination order
    number_of_nodes = len(peo)
    assert number_of_nodes == old_graph.number_of_nodes()

    graph, label_dict = relabel_graph_nodes(
        old_graph, dict(zip(peo, range(1, number_of_nodes+1))))

    # go over nodes and make adjacent all nodes higher in the order
    for node in sorted(graph.nodes):
        neighbors = list(graph[node])
        higher_neighbors = [neighbor for neighbor
                            in neighbors if neighbor > node]

        # form all pairs of higher neighbors
        if len(higher_neighbors) > 1:
            edges = itertools.combinations(higher_neighbors, 2)

            # Do not add edges over existing edges. This is
            # done to work properly with MultiGraphs
            existing_edges = graph.edges(higher_neighbors)
            fillin_edges = [edge for edge
                            in edges if edge not in existing_edges]
        else:
            fillin_edges = None

        # Add edges between all neighbors
        if fillin_edges is not None:
            graph.add_edges_from(
                fillin_edges, tensor=f'E{node}',
                hash_tag=hash((f'E{node}',
                               tuple(neighbors),
                               random.random())))

    # relabel graph back so peo is a correct elimination order
    # of the resulting chordal graph
    graph, _ = relabel_graph_nodes(graph, label_dict)
    return graph


def get_fillin_graph2(old_graph, peo):
    """
    Provided a graph and an order of its indices, returns a
    triangulation of that graph corresponding to the order.

    The algorithm is copied from
    "Simple Linear Time Algorithm To Test Chordality of Graph"
    by R. E. Tarjan and M. Yannakakis

    Parameters
    ----------
    old_graph : nx.Graph or nx.MultiGraph
                graph to triangulate
    peo : elimination order to use for triangulation

    Returns
    -------
    nx.Graph or nx.MultiGraph
                triangulated graph
    """
    number_of_nodes = len(peo)
    graph = copy.deepcopy(old_graph)

    # Safeguard check. May be removed for partial triangulation
    assert number_of_nodes == graph.number_of_nodes()

    index = [0 for ii in range(number_of_nodes)]
    f = [0 for ii in range(number_of_nodes)]

    for ii in range(1, number_of_nodes+1):
        w = peo[ii-1]
        f[w-1] = w
        index[w-1] = ii
        neighbors = list(graph[w])
        lower_neighbors = [v for v in neighbors
                           if peo.index(v)+1 < ii]
        hash_seed = random.random()
        for v in lower_neighbors:
            x = v
            while index[x-1] < ii:
                index[x-1] = ii
                # Check that edge does not exist
                # This may happen if peo is not ordered?
                if (x, w) not in graph.edges(w):
                    graph.add_edge(
                        x, w,
                        tensor=f'E{w}',
                        hash_tag=hash((f'E{w}',
                                       hash_seed)))
                x = f[x-1]
            if f[x-1] == x:
                f[x-1] = w
    return graph


def is_peo_zero_fillin(old_graph, peo):
    """
    Test if the elimination order corresponds to the zero
    fillin of the graph.

    Parameters
    ----------
    graph : nx.Graph or nx.MultiGraph
                triangulated graph to test
    peo : elimination order to use for testing

    Returns
    -------
    bool
            True if elimination order has zero fillin
    """
    # get a copy of graph in the elimination order

    number_of_nodes = len(peo)
    graph, label_dict = relabel_graph_nodes(
        old_graph, dict(zip(peo, range(1, number_of_nodes+1))))

    # go over nodes and make adjacent all nodes higher in the order
    for node in sorted(graph.nodes):
        neighbors = list(graph[node])
        higher_neighbors = [neighbor for neighbor
                            in neighbors if neighbor > node]

        # form all pairs of higher neighbors
        if len(higher_neighbors) > 1:
            edges = itertools.combinations(higher_neighbors, 2)

            # Do not add edges over existing edges. This is
            # done to work properly with MultiGraphs
            existing_edges = graph.edges(higher_neighbors)
            fillin_edges = [edge for edge
                            in edges if edge not in existing_edges]
        else:
            fillin_edges = []

        # Add edges between all neighbors
        if len(fillin_edges) > 0:
            return False
    return True


def is_peo_zero_fillin2(graph, peo):
    """
    Test if the elimination order corresponds to the zero
    fillin of the graph.

    Parameters
    ----------
    graph : nx.Graph or nx.MultiGraph
                triangulated graph to test
    peo : elimination order to use for testing

    Returns
    -------
    bool
            True if elimination order has zero fillin
    """
    number_of_nodes = len(peo)

    index = [0 for ii in range(number_of_nodes)]
    f = [0 for ii in range(number_of_nodes)]

    for ii in range(1, number_of_nodes+1):
        w = peo[ii-1]
        f[w-1] = w
        index[w-1] = ii
        neighbors = list(graph[w])
        lower_neighbors = [v for v in neighbors
                           if peo.index(v)+1 < ii]
        for v in lower_neighbors:
            index[v-1] = ii
            if f[v-1] == v:
                f[v-1] = w
        for v in lower_neighbors:
            if index[f[v-1] - 1] < ii:
                return False
    return True


def is_clique(old_graph, vertices):
    """
    Tests if vertices induce a clique in the graph
    Multigraphs are reduced to normal graphs

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
          graph
    vertices : list
          vertices which are tested
    Returns
    -------
    bool
        True if vertices induce a clique
    """
    subgraph = old_graph.subgraph(vertices)

    # Remove selfloops so the clique is well defined
    have_edges = set(subgraph.edges()) - set(subgraph.selfloop_edges())

    # Sort all edges to be in the (low, up) order
    have_edges = set([tuple(sorted(edge)) for edge in have_edges])

    want_edges = set([
        tuple(sorted(edge))
        for edge in itertools.combinations(vertices, 2)
    ])
    return want_edges == have_edges


def maximum_cardinality_search(
        old_graph, last_clique_vertices=[]):
    """
    This function builds elimination order of a chordal graph
    using maximum cardinality search algorithm.
    If last_clique_vertices is
    provided the algorithm will place these indices at the end
    of the elimination list in the same order as provided.

    Parameters
    ----------
    graph : nx.Graph or nx.MultiGraph
            chordal graph to build the elimination order
    last_clique_vertices : list, default []
            list of vertices to be placed at the end of
            the elimination order
    Returns
    -------
    list
        Perfect elimination order
    """

    # Check is last_clique_vertices is a clique

    graph = copy.deepcopy(old_graph)
    n_nodes = graph.number_of_nodes()

    nodes_by_ordered_neighbors = [[] for ii in range(0, n_nodes)]
    for node in graph.nodes:
        graph.node[node]['n_ordered_neighbors'] = 0
        nodes_by_ordered_neighbors[0].append(node)

    last_nonempty = 0
    peo = []

    for ii in range(n_nodes, 0, -1):
        # Take any unordered node with highest cardinality
        # or the ones in the last_clique_vertices if it was provided

        if len(last_clique_vertices) > 0:
            # Forcibly select the node from the clique
            node = last_clique_vertices.pop()
            # The following should always be possible if
            # last_clique_vertices induces a clique and I understood
            # the theorem correctly. If it raises something is wrong
            # with the algorithm/input is not a clique
            try:
                nodes_by_ordered_neighbors[last_nonempty].remove(node)
            except ValueError:
                if not is_clique(graph, last_clique_vertices):
                    raise ValueError(
                        'last_clique_vertices are not a clique')
                else:
                    raise AssertionError('Algorithmic error. Investigate')
        else:
            node = nodes_by_ordered_neighbors[last_nonempty].pop()

        peo = [node] + peo
        graph.node[node]['n_ordered_neighbors'] = -1

        unordered_neighbors = [
            (neighbor, graph.node[neighbor]['n_ordered_neighbors'])
            for neighbor in graph[node]
            if graph.node[neighbor]['n_ordered_neighbors'] >= 0]

        # Increase number of ordered neighbors for all adjacent
        # unordered nodes
        for neighbor, n_ordered_neighbors in unordered_neighbors:
            nodes_by_ordered_neighbors[n_ordered_neighbors].remove(
                neighbor)
            graph.node[neighbor][
                'n_ordered_neighbors'] = n_ordered_neighbors + 1
            nodes_by_ordered_neighbors[n_ordered_neighbors + 1].append(
                neighbor)

        last_nonempty += 1
        while last_nonempty >= 0:
            if len(nodes_by_ordered_neighbors[last_nonempty]) == 0:
                last_nonempty -= 1
            else:
                break

    return peo


def get_equivalent_peo(old_graph, peo, clique_vertices):
    """
    This function returns an equivalent peo with
    the clique_indices in the rest of the new order
    """
    # Ensure that the graph is simple
    graph = get_simple_graph(old_graph)

    # Complete the graph
    graph_chordal = get_fillin_graph(graph, peo)

    # MCS will produce alternative PEO with this clique at the end
    new_peo = maximum_cardinality_search(graph_chordal,
                                         list(clique_vertices))

    return new_peo


def get_equivalent_peo_naive(peo, clique_vertices):
    """
    This function returns an equivalent peo with
    the clique_indices in the rest of the new order
    """
    new_peo = copy.deepcopy(peo)
    for node in clique_vertices:
        new_peo.remove(node)

    new_peo = new_peo + clique_vertices
    return new_peo


def get_node_min_fill_heuristic(graph):
    """
    Calculates the next node for the min-fill
    heuristic, as described in V. Gogate and R.  Dechter
    :url:`http://arxiv.org/abs/1207.4109`

    Parameters
    ----------
    graph : networkx.Graph graph to estimate

    Returns
    -------
    node : node-type
           node with minimal degree
    degree : int
           degree of the node
    """
    min_fill = np.inf

    for node in graph.nodes:
        neighbors_g = graph.subgraph(
            graph.neighbors(node))
        degree = neighbors_g.number_of_nodes()
        n_edges_filled = neighbors_g.number_of_edges()

        # All possible edges without selfloops
        n_edges_max = int(degree*(degree-1) // 2)
        fill = n_edges_max - n_edges_filled
        if fill <= min_fill:
            min_fill_node = node
            min_fill_degree = degree
        min_fill = min(fill, min_fill)

    return min_fill_node, min_fill_degree


def get_upper_bound_peo(old_graph,
                        node_heuristic_fn=get_node_min_fill_heuristic):
    """
    Calculates an upper bound on treewidth using one of the
    heuristics given by the node_heuristic_fn.
    Best is min-fill,
    as described in V. Gogate and R. Dechter
    :url:`http://arxiv.org/abs/1207.4109`

    Parameters
    ----------
    graph : networkx.Graph
           graph to estimate
    node_heuristic_fn : function
           heuristic function, default min_fill_node

    Returns
    -------
    peo : list
           list of nodes in perfect elimination order
    treewidth : int
           treewidth corresponding to peo
    """
    graph = copy.deepcopy(old_graph)

    node, max_degree = node_heuristic_fn(graph)
    peo = [node]
    eliminate_node(graph, node, self_loops=False)

    for ii in range(graph.number_of_nodes()):
        node, degree = node_heuristic_fn(graph)
        peo.append(node)
        max_degree = max(max_degree, degree)
        eliminate_node(graph, node, self_loops=False)

    return peo, max_degree  # this is clique size - 1


def test_get_fillin_graph():
    """
    Test graph filling using the elimination order
    """
    import time
    nq, g = read_graph(
        'test_circuits/inst/cz_v2/10x10/inst_10x10_60_1.txt')

    tim1 = time.time()
    g1 = get_fillin_graph(g, list(range(1, g.number_of_nodes() + 1)))
    tim2 = time.time()
    g2 = get_fillin_graph(g, list(range(1, g.number_of_nodes() + 1)))
    tim3 = time.time()

    assert nx.is_isomorphic(g1, g2)
    print(tim2 - tim1, tim3 - tim2)


def test_is_zero_fillin():
    """
    Test graph filling using the elimination order
    """
    import time
    nq, g = read_graph(
        'test_circuits/inst/cz_v2/10x10/inst_10x10_60_1.txt')

    g1 = get_fillin_graph(g, list(range(1, g.number_of_nodes() + 1)))

    tim1 = time.time()
    print(
        is_peo_zero_fillin(g1, list(range(1, g.number_of_nodes() + 1))))
    tim2 = time.time()
    print(
        is_peo_zero_fillin2(g1, list(range(1, g.number_of_nodes() + 1))))
    tim3 = time.time()

    print(tim2 - tim1, tim3 - tim2)


def test_maximum_cardinality_search():
    """Test maximum cardinality search algorithm"""

    # Read graph
    nq, old_g = read_graph('inst_2x2_7_0.txt')

    # Make random clique
    vertices = list(np.random.choice(old_g.nodes, 4, replace=False))
    while is_clique(old_g, vertices):
        vertices = list(np.random.choice(old_g.nodes, 4, replace=False))

    log.info(f"Clique on vertices: {vertices}")
    g = make_clique_on(old_g, vertices)

    # Make graph completion
    peo, tw = get_peo(g)
    g_chordal = get_fillin_graph(g, peo)

    # MCS will produce alternative PEO with this clique at the end
    new_peo = maximum_cardinality_search(g_chordal, list(vertices))

    # Test if new peo is correct
    assert is_peo_zero_fillin(g_chordal, peo)
    assert is_peo_zero_fillin(g_chordal, new_peo)
    new_tw = get_treewidth_from_peo(g, new_peo)
    assert tw == new_tw

    print('peo:', peo)
    print('new_peo:', new_peo)


def test_is_clique():
    """Test is_clique"""
    nq, g = read_graph('inst_2x2_7_0.txt')

    # select some random vertices
    vertices = list(np.random.choice(g.nodes, 4, replace=False))
    while is_clique(g, vertices):
        vertices = list(np.random.choice(g.nodes, 4, replace=False))

    g_new = make_clique_on(g, vertices)

    assert is_clique(g_new, vertices)
