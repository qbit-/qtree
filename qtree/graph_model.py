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

import qtree.system_defs as defs
import qtree.utils as utils

from qtree.optimizer import Var, Tensor
from qtree.quickbb_api import gen_cnf, run_quickbb
from qtree.logger_setup import log

random.seed(0)


def circ2graph(qubit_count, circuit, max_depth=None,
               omit_terminals=True):
    """
    Constructs a graph from a circuit in the form of a
    list of lists.

    Parameters
    ----------
    qubit_count : int
            number of qubits in the circuit
    circuit : list of lists
            quantum circuit as returned by
            :py:meth:`operators.read_circuit_file`
    max_depth : int, default None
            Maximal depth of the circuit which should be used
    omit_terminals : bool, default True
            If terminal nodes should be excluded from the final
            graph.

    Returns
    -------
    graph : networkx.MultiGraph
            Graph which corresponds to the circuit
    """
    import functools
    import qtree.operators as ops

    if max_depth is None:
        max_depth = len(circuit)

    # Let's build the graph.
    # The circuit is built from left to right, as it operates
    # on the ket ( |0> ) from the left. We thus first place
    # the bra ( <x| ) and then put gates in the reverse order

    # Fill the variable `frame`
    layer_variables = list(range(qubit_count))
    current_var_idx = qubit_count

    # Initialize the graph
    graph = nx.MultiGraph()

    # Populate nodes and save variables of the bra
    bra_variables = []
    for var in layer_variables:
        graph.add_node(var, name=f'o_{var}', size=2)
        bra_variables.append(Var(var, name=f"o_{var}"))

    # Place safeguard measurement circuits before and after
    # the circuit
    measurement_circ = [[ops.M(qubit) for qubit in range(qubit_count)]]

    combined_circ = functools.reduce(
        lambda x, y: itertools.chain(x, y),
        [measurement_circ, reversed(circuit[:max_depth])])

    # Start building the graph in reverse order
    for layer in combined_circ:
        for op in layer:
            # build the indices of the gate. If gate
            # changes the basis of a qubit, a new variable
            # has to be introduced and current_var_idx is increased.
            # The order of indices
            # is always (a_new, a, b_new, b, ...), as
            # this is how gate tensors are chosen to be stored
            variables = []
            current_var_idx_copy = current_var_idx
            for qubit in op.qubits:
                if qubit in op.changed_qubits:
                    variables.extend(
                        [layer_variables[qubit],
                         current_var_idx_copy])
                    graph.add_node(
                        current_var_idx_copy,
                        name='v_{}'.format(current_var_idx_copy),
                        size=2)
                    current_var_idx_copy += 1
                else:
                    variables.extend([layer_variables[qubit]])
            # Form a tensor and add a clique to the graph
            tensor = {'name': op.name, 'indices': tuple(variables),
                      'data_key': op.data_key}

            if len(variables) > 1:
                edges = itertools.combinations(variables, 2)
            else:
                edges = [(variables[0], variables[0])]

            graph.add_edges_from(edges, tensor=tensor)

            # Update current variable frame
            for qubit in op.changed_qubits:
                layer_variables[qubit] = current_var_idx
                current_var_idx += 1

    # Finally go over the qubits, append measurement gates
    # and collect ket variables
    ket_variables = []

    op = ops.M(0)  # create a single measurement gate object

    for qubit in range(qubit_count):
        var = layer_variables[qubit]
        new_var = current_var_idx

        ket_variables.append(Var(new_var, name=f'i_{qubit}', size=2))
        # update graph and variable `frame`
        graph.add_node(new_var, name=f'i_{qubit}', size=2)
        tensor = {'name': op.name, 'indices': (var, new_var),
                  'data_key': op.data_key}

        graph.add_edge(var, new_var, tensor=tensor)
        layer_variables[qubit] = new_var
        current_var_idx += 1

    if omit_terminals:
        graph.remove_nodes_from(
            tuple(map(int, bra_variables + ket_variables)))

    v = graph.number_of_nodes()
    e = graph.number_of_edges()

    log.info(f"Generated graph with {v} nodes and {e} edges")
    # log.info(f"last index contains from {layer_variables}")

    return graph


def buckets2graph(buckets, ignore_variables=[]):
    """
    Takes buckets and produces a corresponding undirected graph. Single
    variable tensors are coded as self loops and there may be
    multiple parallel edges.

    Parameters
    ----------
    buckets : list of lists
    ignore_variables : list, optional
       Variables to be deleted from the resulting graph.
       Numbering is 0-based.

    Returns
    -------
    graph : networkx.MultiGraph
            contraction graph of the circuit
    """
    # convert everything to int
    ignore_variables = [int(var) for var in ignore_variables]

    graph = nx.MultiGraph()

    # Let's build an undirected graph for variables
    for n, bucket in enumerate(buckets):
        for tensor in bucket:
            new_nodes = []
            for idx in tensor.indices:
                # This may reintroduce the same node many times,
                # be careful if using something other than
                graph.add_node(int(idx), name=idx.name, size=idx.size)
                new_nodes.append(int(idx))
            if len(new_nodes) > 1:
                edges = itertools.combinations(new_nodes, 2)
            else:
                # If this is a single variable tensor, add self loop
                node = new_nodes[0]
                edges = [[node, node]]
            graph.add_edges_from(
                edges,
                tensor={
                    'name': tensor.name,
                    'indices': tuple(map(int, tensor.indices)),
                    'data_key': tensor.data_key
                    }
            )

    # Delete any requested variables from the final graph
    if len(ignore_variables) > 0:
        for var in ignore_variables:
            remove_node(graph, var)

    return graph


def relabel_graph_nodes(graph, label_dict=None, with_data=True):
    """
    Relabel graph nodes.The graph
    is relabelled (and returned) according to the label
    dictionary and an inverted dictionary is returned.
    Only integers are allowed as labels. If some other
    objects will be passed inn the label_dict, they will
    be attempted to convert to integers. If no dictionary
    will be passed then nodes will be relabeled according to
    consequtive integers starting from 0.

    In contrast to the Networkx version this one also relabels
    indices in the 'tensor' parameter of edges

    Parameters
    ----------
    graph : networkx.Graph
            graph to relabel
    label_dict : dict-like, default None
            dictionary for relabelling {old : new}
    with_data : bool, default True
            if we will check and relabel data on the edges as well
    Returns
    -------
    new_graph : networkx.Graph
            relabeled graph
    label_dict : dict
            {new : old} dictionary for inverse relabeling
    """
    # Ensure label dictionary contains integers or create one
    if label_dict is None:
        label_dict = {int(old): num for num, old in
                      enumerate(graph.nodes(data=False))}
    else:
        label_dict = {int(key): int(val)
                      for key, val in label_dict.items()}

    tensors_hash_table = {}

    # make a deep copy. We want to change all attributes without
    # interference
    new_graph = copy.deepcopy(graph)

    if with_data:
        args_to_nx = {'data': 'tensor'}
        if graph.is_multigraph():
            args_to_nx['keys'] = True

        for edgedata in graph.edges.data(**args_to_nx):
            *edge, tensor = edgedata
            if tensor is not None:
                # create new tensor only if it was not encountered
                key = hash((tensor['data_key'],
                            tensor['indices']))
                if key not in tensors_hash_table:
                    indices = tuple(label_dict[idx]
                                    for idx in tensor['indices'])
                    new_tensor = copy.deepcopy(tensor)
                    new_tensor['indices'] = indices
                    tensors_hash_table[key] = new_tensor
                else:
                    new_tensor = tensors_hash_table[key]
                new_graph.edges[
                    edge]['tensor'] = copy.deepcopy(new_tensor)

    # Then relabel nodes.
    new_graph = nx.relabel_nodes(new_graph, label_dict, copy=True)

    # invert the dictionary
    inv_label_dict = {val: key for key, val in label_dict.items()}

    return new_graph, inv_label_dict


def get_simple_graph(graph, parallel_edges=False, self_loops=False):
    """
    Simplifies graph: MultiGraphs are converted to Graphs,
    selfloops are removed
    """
    if not parallel_edges:
        # deepcopy is critical here to copy edge dictionaries
        graph = nx.Graph(copy.deepcopy(graph), copy=False)
    if not self_loops:
        graph.remove_edges_from(graph.selfloop_edges())

    return graph


def get_peo(old_graph,
            quickbb_extra_args=" --time 60 --min-fill-ordering ",
            input_suffix=None, keep_input=False,
            int_vars=False):
    """
    Calculates the elimination order for an undirected
    graphical model of the circuit.

    Parameters
    ----------
    graph : networkx.Graph
            graph of the undirected graphical model to decompose
    quickbb_extra_args : str, default '--min-fill-ordering --time 60'
             Optional commands to QuickBB.
    input_suffix : str, default None
             Optional suffix to allow parallel execution.
             If None is provided a random suffix is generated
    keep_input : bool, default False
             Whether to keep input files for debugging
    int_vars : bool, default False
             If returned peo should have integers in place of Var objects
    Returns
    -------
    peo_dict : dict
          containing indices in optimal order of elimination. Order
          is in keys, and index objects are in values
    treewidth : int
          treewidth of the decomposition
    """

    # save initial indices to ensure nothing is missed
    initial_indices = old_graph.nodes()

    # Remove selfloops and parallel edges. Critical
    graph = get_simple_graph(old_graph)

    # Relabel graph nodes to consequtive ints
    graph, initial_to_conseq = relabel_graph_nodes(graph)

    # prepare environment
    if input_suffix is None:
        input_suffix = ''.join(str(random.randint(0, 9))
                               for n in range(8))
    cnffile = 'output/quickbb.' + input_suffix + '.cnf'

    if graph.number_of_edges() > 0:
        gen_cnf(cnffile, graph)
        out_bytes = run_quickbb(cnffile, defs.QUICKBB_COMMAND)

        # Extract order
        m = re.search(b'(?P<peo>(\d+ )+).*Treewidth=(?P<treewidth>\s\d+)',
                      out_bytes, flags=re.MULTILINE | re.DOTALL)

        peo = [int(ii) for ii in m['peo'].split()]

        # Map peo back to original indices. PEO in QuickBB is 1-based
        # but we need it 0-based
        peo = [initial_to_conseq[pp - 1] for pp in peo]

        treewidth = int(m['treewidth'])
    else:
        peo = []
        treewidth = 0

    # find the rest of indices which quickBB did not spit out.
    # Those include isolated nodes (don't affect
    # scaling and may be added to the end of the variables list)
    # and something else

    isolated_nodes = nx.isolates(old_graph)
    peo = peo + sorted(isolated_nodes, key=int)

    # assert(set(initial_indices) - set(peo) == set())
    missing_indices = set(initial_indices)-set(peo)
    # The next line needs review. Why quickBB misses some indices?
    # It is here to make program work, but is it an optimal order?
    peo = peo + sorted(list(missing_indices), key=int)

    # Ensure no indices were missed
    assert(sorted(peo, key=int) == sorted(initial_indices, key=int))
    # log.info('Final peo from quickBB:\n{}'.format(peo))

    # remove input file to honor EPA
    if not keep_input:
        try:
            os.remove(cnffile)
        except FileNotFoundError:
            pass

    # transform PEO to a list of Var objects as expected by
    # other parts of code
    if int_vars:
        return peo, treewidth
    else:
        peo_vars = [Var(v, size=old_graph.nodes[v]['size'],
                        name=old_graph.nodes[v]['name']) for v in peo]
        return peo_vars, treewidth


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
    idx_parallel : list of Idx
          variables removed by parallelization
    graph : networkx.Graph
          new graph without parallelized variables
    """
    graph = copy.deepcopy(old_graph)

    indices = [var for var in graph.nodes(data=False)]
    idx_parallel = np.random.choice(
        indices, size=n_var_parallel, replace=False)

    idx_parallel_var = [Var(var, size=graph.nodes[var])
                        for var in idx_parallel]

    for idx in idx_parallel:
        remove_node(graph, idx)

    log.info("Removed indices by parallelization:\n{}".format(idx_parallel))
    log.info("Removed {} variables".format(len(idx_parallel)))
    peo, treewidth = get_peo(graph)

    return sorted(idx_parallel_var, key=int), graph


def get_cost_by_node(graph, node):
    """
    Outputs the cost corresponding to the
    contraction of the node in the graph

    Parameters
    ----------
    graph : networkx.MultiGraph
               Graph containing the information about the contraction

    node : node of the graph (such that graph can be indexed by it)

    Returns
    -------
    memory : int
              Memory cost for contraction of node
    flops : int
              Flop cost for contraction of node
    """
    neighbors_with_size = {neighbor: graph.nodes[neighbor]['size']
                           for neighbor in graph[node]}

    # We have to find all unique tensors which will be contracted
    # in this bucket. They label the edges coming from
    # the current node. Application of identical tensors many times
    # can be encoded in multiple edges between the node and its neighbor.
    # We have to count the number of unique tensors.
    tensors = []
    selfloop_tensors = []

    args_to_nx = {'data': 'tensor'}

    if graph.is_multigraph():
        args_to_nx['keys'] = True

    for edgedata in graph.edges.data(**args_to_nx):
        *edge, tensor = edgedata
        u, v, *edge_key = edge
        edge_key = edge_key[0] if edge_key != [] else 0
        # the tuple (edge_key, indices, data_key) uniquely
        # identifies a tensor
        tensors.append((
            edge_key, tensor['indices'], tensor['data_key']))
        if u == v:
            selfloop_tensors.append((
                edge_key, tensor['indices'], tensor['data_key']))

    # get unique tensors
    tensors = set(tensors)

    # Now find the size of the result.
    # Ensure the node itself from the list of its neighbors.
    # This eliminates possible self loop
    neighbors_wo_node = copy.copy(neighbors_with_size)
    while node in neighbors_wo_node:
        neighbors_wo_node.pop(node)

    # memory estimation: the size of the result + all sizes of terms
    size_of_the_result = np.prod([
        val for val in neighbors_wo_node.values()])
    memory = size_of_the_result
    for tensor_key in tensors:
        _, indices, _ = tensor_key
        mem = np.prod([graph.nodes[idx]['size'] for idx in indices])
        memory += mem

    # Now calculate number of FLOPS
    n_unique_tensors = len(tensors)
    assert n_unique_tensors > 0
    n_multiplications = n_unique_tensors - 1

    # There are n_multiplications and 1 addition
    # repeated size_of_the_result*size_of_contracted_variable
    # times for each contraction
    flops = (size_of_the_result *
             graph.nodes[node]['size']*(1 + n_multiplications))

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
    # Delete node itself from the list of its neighbors.
    # This eliminates possible self loop
    neighbors_wo_node = list(graph[node])
    while node in neighbors_wo_node:
        neighbors_wo_node.remove(node)

    graph.remove_node(node)

    # prune all edges containing the removed node
    edges_to_remove = []
    args_to_nx = {'data': 'tensor', 'nbunch': neighbors_wo_node,
                  'default': {'indices': []}}

    if graph.is_multigraph():
        args_to_nx['keys'] = True

    for edgedata in graph.edges.data(**args_to_nx):
        *edge, tensor = edgedata
        if node in tensor['indices']:
            edges_to_remove.append(edge)

    graph.remove_edges_from(edges_to_remove)

    # prepare new tensor
    if len(neighbors_wo_node) > 1:
        edges = itertools.combinations(neighbors_wo_node, 2)
    elif len(neighbors_wo_node) == 1 and self_loops:
        # This node had a single neighbor, add self loop to it
        edges = [[neighbors_wo_node[0], neighbors_wo_node[0]]]
    else:
        # This node had no neighbors
        edges = None

    if edges is not None:
        graph.add_edges_from(
            edges,
            tensor={
                'name': 'E{}'.format(int(node)),
                'indices': tuple(neighbors_wo_node),
                'data_key':  None
            }
        )


def remove_node(graph, node, self_loops=True):
    """
    Eliminates node if its value was fixed

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
    # Delete node itself from the list of its neighbors.
    # This eliminates possible self loop
    neighbors_wo_node = list(graph[node])
    while node in neighbors_wo_node:
        neighbors_wo_node.remove(node)

    # prune all tensors containing the removed node
    args_to_nx = {'data': 'tensor', 'nbunch': neighbors_wo_node,
                  'default': {'indices': []}}
    if graph.is_multigraph():
        args_to_nx['keys'] = True

    new_selfloops = []
    for edgedata in graph.edges.data(**args_to_nx):
        *edge, tensor = edgedata
        indices = tensor['indices']
        if node in indices:
            new_indices = tuple(idx for idx in indices if idx != node)
            tensor['indices'] = new_indices
            # Invalidate data pointer as this tensor is a slice
            tensor['data_key'] = None
            if self_loops and len(new_indices) == 1:  # create a self loop
                neighbor = new_indices[0]
                new_selfloops.append((neighbor, tensor))
            else:
                graph.edges[edge]['tensor'] = tensor

    graph.remove_node(node)

    # introduce selfloops
    if self_loops:
        for v, tensor in new_selfloops:
            graph.add_edge(v, v, tensor=tensor)


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
    # We have to find all unique tensors which will be contracted
    # in this bucket. They label the edges coming from
    # the current node. Application of identical tensors many times
    # can be encoded in multiple edges between the node and its neighbor.
    # We have to count the number of unique tensors.
    tensors = []
    args_to_nx = {'data': 'tensor'}

    if graph.is_multigraph():
        args_to_nx['keys'] = True

    for edgedata in graph.edges.data(**args_to_nx):
        *edge, tensor = edgedata
        u, v, *edge_key = edge
        edge_key = edge_key[0] if edge_key != [] else 0
        # the tuple (edge_key, indices, data_key) uniquely
        # identifies a tensor
        tensors.append((
            edge_key, tensor['indices'], tensor['data_key']))

    # get unique tensors
    tensors = set(tensors)

    # memory estimation
    memory = 0
    for tensor_key in tensors:
        _, indices, _ = tensor_key
        mem = np.prod([graph.nodes[idx]['size'] for idx in indices])
        memory += mem

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
    nodes = sorted(graph.nodes, key=int)
    free_vars = [int(var) for var in free_vars]

    # Early return if graph is empty
    if len(nodes) == 0:
        return [1], [1]

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
    provided graph. The graph is **ASSUMED** to be in the optimal
    elimination order, e.g. the nodes have to be relabelled by
    peo

    Parameters
    ----------
    graph : networkx.Graph without self-loops and parallel edges

    Returns
    -------
    nodes_by_mem_reduction : dict
    """

    graph = copy.deepcopy(old_graph)

    # Get flop cost of the bucket elimination
    initial_mem, initial_flop = cost_estimator(graph)

    nodes_by_mem_reduction = []
    for node in graph.nodes(data=False):
        reduced_graph = copy.deepcopy(graph)
        # Take out one node
        remove_node(reduced_graph, node)
        mem, flop = cost_estimator(reduced_graph)
        delta = sum(initial_mem) - sum(mem)

        nodes_by_mem_reduction.append((node, delta))

    return nodes_by_mem_reduction


def get_node_by_treewidth_reduction(old_graph):
    """
    Returns a list of pairs (node : reduction_in_treewidth) for the
    provided graph. The graph is **ASSUMED** to be in the optimal
    elimination order, e.g. the nodes have to be relabelled by
    peo

    Parameters
    ----------
    graph : networkx.Graph without self-loops and parallel edges

    Returns
    -------
    nodes_by_treewidth_reduction : dict
    """
    number_of_nodes = old_graph.number_of_nodes()
    graph = copy.deepcopy(old_graph)

    # Get flop cost of the bucket elimination
    initial_treewidth = get_treewidth_from_peo(
        graph, list(range(number_of_nodes)))

    nodes_by_treewidth_reduction = []
    for node in graph.nodes(data=False):
        reduced_graph = copy.deepcopy(graph)
        # Take out one node
        remove_node(reduced_graph, node)
        # Renumerate graph nodes to be consequtive ints (may be redundant)
        order = (list(range(node))
                 + list(range(node + 1, number_of_nodes)))
        reduced_graph, _ = relabel_graph_nodes(
            reduced_graph, dict(zip(order, range(number_of_nodes-1)))
        )
        treewidth = get_treewidth_from_peo(
            reduced_graph, list(range(number_of_nodes-1)))
        delta = initial_treewidth - treewidth

        nodes_by_treewidth_reduction.append((node, delta))

    return nodes_by_treewidth_reduction


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
    # graph = get_simple_graph(old_graph)
    # import pdb
    # pdb.set_trace()
    graph = copy.deepcopy(old_graph)

    # convert everything to int
    forbidden_nodes = [int(var) for var in forbidden_nodes]

    # get nodes by metric in descending order
    nodes_by_metric = metric_fn(graph)
    nodes_by_metric.sort(key=lambda pair: int(pair[1]), reverse=True)

    nodes_by_metric_allowed = []
    for node, metric in nodes_by_metric:
        if node not in forbidden_nodes:
            nodes_by_metric_allowed.append((node, metric))

    idx_parallel = []
    for ii in range(n_var_parallel):
        node, metric = nodes_by_metric_allowed[ii]
        idx_parallel.append(node)

    # create var objects from nodes
    idx_parallel_var = [Var(var, size=graph.nodes[var]['size'])
                        for var in idx_parallel]

    for idx in idx_parallel:
        remove_node(graph, idx)

    log.info("Removed indices by parallelization:\n{}".format(idx_parallel))
    log.info("Removed {} variables".format(len(idx_parallel)))

    return idx_parallel_var, graph


def split_graph_with_mem_constraint_greedy(
        old_graph,
        n_var_parallel_min=0,
        mem_constraint=defs.MAXIMAL_MEMORY,
        step_by=5,
        n_var_parallel_max=None,
        metric_fn=get_node_by_mem_reduction,
        forbidden_nodes=[]):
    """
    This function splits graph by greedily selecting next nodes
    up to the n_var_parallel
    using the metric function and recomputing PEO after
    each node elimination. The graph is **ASSUMED** to be in
    the perfect elimination order

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
    graph : networkx.Graph
             reduced contraction graph
    """
    # convert everything to int
    forbidden_nodes = [int(var) for var in forbidden_nodes]

    graph = copy.deepcopy(old_graph)
    n_var_total = old_graph.number_of_nodes()
    if n_var_parallel_max is None:
        n_var_parallel_max = n_var_total

    mem_cost, flop_cost = cost_estimator(graph)
    max_mem = sum(mem_cost)

    idx_parallel = []
    idx_parallel_var = []
    for n_var_parallel in range(0, n_var_parallel_max, step_by):
        # Get optimal order
        peo, tw = get_peo(graph)
        graph_optimal, inverse_order = relabel_graph_nodes(
            graph, dict(zip(peo, range(len(peo)))))

        # get nodes by metric in descending order
        nodes_by_metric_optimal = metric_fn(graph_optimal)
        nodes_by_metric_optimal.sort(
            key=lambda pair: pair[1], reverse=True)

        nodes_by_metric_allowed = []
        for node, metric in nodes_by_metric_optimal:
            if inverse_order[node] not in forbidden_nodes:
                nodes_by_metric_allowed.append(
                    (inverse_order[node], metric))

        # Take first nodes by cost and map them back to original
        # order
        nodes, costs = zip(
            *nodes_by_metric_allowed[:step_by])

        # Update list and update graph
        idx_parallel += nodes

        # create var objects from nodes
        idx_parallel_var += [Var(var, size=graph.nodes[var]['size'])
                             for var in nodes]

        for node in nodes:
            remove_node(graph, node)

        # Renumerate graph nodes to be consequtive ints (may be redundant)
        label_dict = dict(zip(sorted(graph.nodes),
                              range(len(graph.nodes()))))

        graph_relabelled, _ = relabel_graph_nodes(graph, label_dict)
        mem_cost, flop_cost = cost_estimator(graph_relabelled)

        max_mem = sum(mem_cost)

        if (max_mem <= mem_constraint
           and n_var_parallel >= n_var_parallel_min):
            break

    if max_mem > mem_constraint:
        raise ValueError('Maximal memory constraint is not met')

    return idx_parallel_var, graph


def split_graph_by_metric_greedy(
        old_graph, n_var_parallel=0,
        metric_fn=get_node_by_treewidth_reduction,
        greedy_step_by=1, forbidden_nodes=[]):
    """
    This function splits graph by greedily selecting next nodes
    up to the n_var_parallel
    using the metric function and recomputing PEO after
    each node elimination

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
                Default get_node_by_mem_reduction
    greedy_step_by : int, default 1
                Step size for the greedy algorithm

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
    # import pdb
    # pdb.set_trace()

    # convert everything to int
    forbidden_nodes = [int(var) for var in forbidden_nodes]

    # Simplify graph
    graph = get_simple_graph(old_graph)

    idx_parallel = []
    idx_parallel_var = []
    for ii in range(0, n_var_parallel, greedy_step_by):
        # Get optimal order
        peo, tw = get_peo(graph)
        graph_optimal, inverse_order = relabel_graph_nodes(
            graph, dict(zip(peo, sorted(graph.nodes))))

        # get nodes by metric in descending order
        nodes_by_metric_optimal = metric_fn(graph_optimal)
        nodes_by_metric_optimal.sort(
            key=lambda pair: pair[1], reverse=True)

        nodes_by_metric_allowed = []
        for node, metric in nodes_by_metric_optimal:
            if inverse_order[node] not in forbidden_nodes:
                nodes_by_metric_allowed.append(
                    (inverse_order[node], metric))

        # Take first nodes by cost and map them back to original
        # order
        nodes, costs = zip(
            *nodes_by_metric_allowed[:greedy_step_by])

        # Update list and update graph
        idx_parallel += nodes
        # create var objects from nodes
        idx_parallel_var += [Var(var, size=graph.nodes[var]['size'])
                             for var in nodes]
        for node in nodes:
            remove_node(graph, node)

    return idx_parallel_var, graph


def draw_graph(graph, filename=''):
    """
    Draws graph with spectral layout
    Parameters
    ----------
    graph : networkx.Graph
            graph to draw
    filename : str, default ''
            filename for image output.
            If empty string is passed the graph is displayed
    """
    plt.figure(figsize=(10, 10))
    # pos = nx.spectral_layout(graph)
    pos = nx.spectral_layout(graph)
    nx.draw(graph, pos,
            node_color=(list(map(int, graph.nodes()))),
            node_size=100,
            cmap=plt.cm.Blues,
            with_labels=True,
    )
    if len(filename) == 0:
        plt.show()
    else:
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
    graph_type = type(graph)

    # relabel nodes starting to integers
    label_dict = dict(zip(
        list(sorted(graph.nodes)),
        range(graph.number_of_nodes())
    ))

    # Add size to nodes
    for node in graph.nodes:
        graph.nodes[node]['size'] = 2

    # Add tensors to edges and ensure the graph is a Multigraph
    new_graph = graph_type(
        nx.relabel_nodes(graph, label_dict, copy=True)
        )

    params = {'keys': True} if graph.is_multigraph() else dict()

    for edge in new_graph.edges(**params):
        new_graph.edges[edge].update(
            {'tensor':
             {'name': 'W', 'indices': tuple(edge), 'data_key': None}})

    node_names = dict((node, f'v_{node}') for node in new_graph.nodes)
    nx.set_node_attributes(new_graph, node_names, name='name')

    return new_graph


def generate_erdos_graph(n_nodes, probability):
    """
    Generates a random graph with n_nodes and the probability of
    edge equal probability.

    Parameters
    ----------
    n_nodes : int
          Number of nodes
    probability : float
          probability of edge
    Returns
    -------
    graph : networkx.Graph
          Random graph usable by graph_models
    """

    return wrap_general_graph_for_qtree(
        nx.generators.fast_gnp_random_graph(
            n_nodes,
            probability))


def generate_grid_graph(m, n, periodic=False):
    """
    Generates a 2d grid with possible periodic boundary
    Parameters
    ----------
    m, n: int
          Grid size
    periodic: bool, default False
          If the grid should be made periodic
    """
    return wrap_general_graph_for_qtree(
        nx.generators.grid_2d_graph(m, n, periodic=periodic))


def prune_k_tree(old_ktree, probability, n_cliques=1):
    """
    Prunes a k-tree preserving its treewidth (k).
    The edges are preserved with a given probability.
    The resulting graph is a union of a clique/partial k-trees
    with an Erdos graph

    Parameters
    ----------
    old_ktree: nx.Graph
               This is a ktree to prune
    probability: float
               Probability to preserve edge
    n_cliques: int, default 1
               The number of cliques to save in the result
    Returns
    -------
    pruned_ktree: nx.Graph
    """

    # save a copy to make this function pure
    ktree = copy.deepcopy(old_ktree)

    # choose some cliques to keep. We choose clique roots at random
    preserved_roots = list(np.random.choice(
        ktree.nodes, n_cliques, replace=False))
    # now extract other nodes in cliques which have these roots
    all_neighbors = []
    for root in preserved_roots:
        all_neighbors += list(ktree.neighbors(root))
    all_neighbors = list(set(all_neighbors))

    # finally extract chosen cliques/ktrees
    preserved_subgraph = nx.subgraph(ktree, all_neighbors+preserved_roots)

    # Extract edges to keep
    preserved_edges = sorted(preserved_subgraph.edges())

    # now start pruning the graph
    remove_edges = []
    for edge in ktree.edges:
        keep_edge = bool(np.random.binomial(1, probability))
        if (not keep_edge) and (edge not in preserved_edges):
            remove_edges.append(edge)
        else:
            continue
    ktree.remove_edges_from(remove_edges)

    return ktree


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
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))

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
            graph.add_edges_from(edges)

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
    clique_nodes = tuple(int(var) for var in clique_nodes)
    graph = copy.deepcopy(old_graph)

    if len(clique_nodes) == 0:
        return graph

    edges = [tuple(sorted(edge)) for edge in
             itertools.combinations(clique_nodes, 2)]
    node_idx = min(clique_nodes)
    graph.add_edges_from(edges,
                         tensor={'name': name_prefix + f'{node_idx}',
                                 'indices': clique_nodes,
                                 'data_key': None}
    )
    clique_size = len(clique_nodes)
    log.info(f"Clique of size {clique_size} on vertices: {clique_nodes}")

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
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))

    # get a copy of graph in the elimination order. We do not relabel
    # tensor parameters of edges as it takes too much time
    number_of_nodes = len(peo)
    assert number_of_nodes == old_graph.number_of_nodes()
    graph, inv_label_dict = relabel_graph_nodes(
        old_graph,
        dict(zip(peo, sorted(old_graph.nodes))),
        with_data=False)

    # go over nodes and make adjacent all nodes higher in the order
    for node in sorted(graph.nodes):
        neighbors = list(graph[node])
        higher_neighbors = [neighbor for neighbor in neighbors
                            if neighbor > node]

        # form all pairs of higher neighbors
        if len(higher_neighbors) > 1:
            edges = itertools.combinations(higher_neighbors, 2)

            existing_edges = graph.edges(higher_neighbors, data=False)
            # Do not add edges over existing edges. This is
            # done to work properly with MultiGraphs
            fillin_edges = [edge for edge
                            in edges if edge not in existing_edges]
        else:
            fillin_edges = None

        # Add edges between all neighbors
        if fillin_edges is not None:
            tensor = {'name': 'C{}'.format(node),
                      'indices': (node,) + tuple(neighbors),
                      'data_key': None}
            graph.add_edges_from(
                fillin_edges, tensor=tensor
            )

    # relabel graph back so peo is a correct elimination order
    # of the resulting chordal graph
    graph, _ = relabel_graph_nodes(
        graph, inv_label_dict, with_data=False)

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
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))
    peo_to_conseq = dict(zip(peo, range(len(peo))))

    number_of_nodes = len(peo)
    graph = copy.deepcopy(old_graph)

    # Safeguard check. May be removed for partial triangulation
    assert number_of_nodes == graph.number_of_nodes()

    index = [0 for ii in range(number_of_nodes)]
    f = [0 for ii in range(number_of_nodes)]

    for ii in range(number_of_nodes):
        w = peo[ii]
        idx_w = peo_to_conseq[w]
        f[idx_w] = w
        index[idx_w] = ii
        neighbors = list(graph[w])
        lower_neighbors = [v for v in neighbors
                           if peo.index(v) < ii]
        for v in lower_neighbors:
            x = v
            idx_x = peo_to_conseq[x]
            while index[idx_x] < ii:
                index[idx_x] = ii
                # Check that edge does not exist
                # Tensors added here may not correspond to cliques!
                # Their names are made incompatible with Tensorflow
                # to highlight it
                if (x, w) not in graph.edges(w):
                    tensor = {'name': 'C{}'.format(w),
                              'indices': (w, ) + tuple(neighbors),
                              'data_key': None}
                    graph.add_edge(
                        x, w,
                        tensor=tensor)
                x = f[idx_x]
                idx_x = peo_to_conseq[x]
            if f[idx_x] == x:
                f[idx_x] = w
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
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))

    # get a copy of graph in the elimination order
    graph, label_dict = relabel_graph_nodes(
        old_graph, dict(zip(peo, sorted(old_graph.nodes())))
        )

    # go over nodes and make adjacent all nodes higher in the order
    for node in sorted(graph.nodes):
        neighbors = list(graph[node])
        higher_neighbors = [neighbor for neighbor
                            in neighbors
                            if neighbor > node]

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
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))
    peo_to_conseq = dict(zip(peo, range(len(peo))))

    number_of_nodes = len(peo)

    index = [0 for ii in range(number_of_nodes)]
    f = [0 for ii in range(number_of_nodes)]

    for ii in range(number_of_nodes):
        w = peo[ii]
        idx_w = peo_to_conseq[w]
        f[idx_w] = w
        index[idx_w] = ii
        neighbors = list(graph[w])
        lower_neighbors = [v for v in neighbors
                           if peo.index(v) < ii]
        for v in lower_neighbors:
            idx_v = peo_to_conseq[v]
            index[idx_v] = ii
            if f[idx_v] == v:
                f[idx_v] = w
        for v in lower_neighbors:
            idx_v = peo_to_conseq[v]
            if index[f[idx_v]] < ii:
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
    have_edges = set([tuple(sorted(edge, key=int))
                      for edge in have_edges])

    want_edges = set([
        tuple(sorted(edge, key=int))
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
    # convert input to int
    last_clique_vertices = [int(var) for var in last_clique_vertices]

    # Check is last_clique_vertices is a clique

    graph = copy.deepcopy(old_graph)
    n_nodes = graph.number_of_nodes()

    nodes_number_of_ord_neighbors = {node: 0 for node in graph.nodes}
    # range(0, n_nodes + 1) is important here as we need n+1 lists
    # to ensure proper indexing in the case of a clique
    nodes_by_ordered_neighbors = [[] for ii in range(0, n_nodes + 1)]
    for node in graph.nodes:
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
        nodes_number_of_ord_neighbors[node] = -1

        unordered_neighbors = [
            (neighbor, nodes_number_of_ord_neighbors[neighbor])
            for neighbor in graph[node]
            if nodes_number_of_ord_neighbors[neighbor] >= 0]

        # Increase number of ordered neighbors for all adjacent
        # unordered nodes
        for neighbor, n_ordered_neighbors in unordered_neighbors:
            nodes_by_ordered_neighbors[n_ordered_neighbors].remove(
                neighbor)
            nodes_number_of_ord_neighbors[neighbor] = (
                n_ordered_neighbors + 1)
            nodes_by_ordered_neighbors[n_ordered_neighbors + 1].append(
                neighbor)

        last_nonempty += 1
        while last_nonempty >= 0:
            if len(nodes_by_ordered_neighbors[last_nonempty]) == 0:
                last_nonempty -= 1
            else:
                break

    # Create Var objects
    peo_vars = [Var(var, size=graph.nodes[var]['size'],
                    name=graph.nodes[var]['name'])
                for var in peo]
    return peo_vars


def get_equivalent_peo(old_graph, peo, clique_vertices):
    """
    This function returns an equivalent peo with
    the clique_indices in the rest of the new order
    """
    # Ensure that the graph is simple
    graph = get_simple_graph(old_graph)

    # Complete the graph
    graph_chordal = get_fillin_graph2(graph, peo)

    # MCS will produce alternative PEO with this clique at the end
    new_peo = maximum_cardinality_search(graph_chordal,
                                         list(clique_vertices))

    return new_peo


def get_equivalent_peo_naive(graph, peo, clique_vertices):
    """
    This function returns an equivalent peo with
    the clique_indices in the rest of the new order
    """
    new_peo = copy.deepcopy(peo)
    for node in clique_vertices:
        new_peo.remove(node)

    new_peo = new_peo + clique_vertices
    return new_peo


def get_node_min_fill_heuristic(graph, randomize=False):
    """
    Calculates the next node for the min-fill
    heuristic, as described in V. Gogate and R.  Dechter
    :url:`http://arxiv.org/abs/1207.4109`

    Parameters
    ----------
    graph : networkx.Graph graph to estimate
    randomize : bool, default False
                if a min fill node is selected at random among
                nodes with the same minimal fill
    Returns
    -------
    node : node-type
           node with minimal fill
    degree : int
           degree of the node
    """
    min_fill = np.inf

    min_fill_nodes = []
    for node in graph.nodes:
        neighbors_g = graph.subgraph(
            graph.neighbors(node))
        degree = neighbors_g.number_of_nodes()
        n_edges_filled = neighbors_g.number_of_edges()

        # All possible edges without selfloops
        n_edges_max = int(degree*(degree-1) // 2)
        fill = n_edges_max - n_edges_filled
        if fill == min_fill:
            min_fill_nodes.append((node, degree))
        elif fill < min_fill:
            min_fill_nodes = [(node, degree)]
            min_fill = fill
        else:
            continue
    # Either choose the node at random among equivalent or use
    # the last one
    if randomize:
        min_fill_nodes_d = dict(min_fill_nodes)
        node = np.random.choice(min_fill_nodes_d)
        degree = min_fill_nodes_d[node]
    else:
        node, degree = min_fill_nodes[-1]
    return node, degree


def get_node_min_degree_heuristic(graph, randomize=False):
    """
    Calculates the next node for the min-degree
    heuristic, as described in V. Gogate and R.  Dechter
    :url:`http://arxiv.org/abs/1207.4109`

    Parameters
    ----------
    graph : networkx.Graph graph to estimate
    randomize : bool, default False
                if a min degree node is selected at random among
                nodes with the same minimal degree

    Returns
    -------
    node : node-type
           node with minimal degree
    degree : int
           degree of the node
    """
    nodes_by_degree = sorted(list(graph.degree()),
                             key=lambda pair: pair[1])
    min_degree = nodes_by_degree[0][1]

    for idx, (node, degree) in enumerate(nodes_by_degree):
        if degree > min_degree:
            break
    min_degree_nodes = nodes_by_degree[:idx]

    # Either choose the node at random among equivalent or use
    # the last one
    if randomize:
        nodes, _ = zip(*min_degree_nodes)
        node = np.random.choice(nodes)
    else:
        node = min_degree_nodes[-1][0]

    return node, min_degree


def get_node_max_cardinality_heuristic(graph, randomize=False):
    """
    Calculates the next node for the maximal cardinality search
    heuristic

    Parameters
    ----------
    graph : networkx.Graph graph to estimate
    randomize : bool, default False
                if a min degree node is selected at random among
                nodes with the same minimal degree

    Returns
    -------
    node : node-type
           node with minimal degree
    degree : int
           degree of the node
    """
    max_cardinality = -1
    max_cardinality_nodes = []

    for node in graph.nodes:
        cardinality = graph.nodes[node].get('cardinality', 0)
        degree = graph.degree(node)
        if cardinality > max_cardinality:
            max_cardinality_nodes = [(node, degree)]
        elif cardinality == max_cardinality:
            max_cardinality_nodes.append((node, degree))
        else:
            continue
    # Either choose the node at random among equivalent or use
    # the last one
    if randomize:
        max_cardinality_nodes_d = dict(max_cardinality_nodes)
        node = np.random.choice(max_cardinality_nodes_d)
        degree = max_cardinality_nodes_d[node]
    else:
        node, degree = max_cardinality_nodes[-1]

    # update the graph to hold the cardinality information
    for neighbor in graph.neighbors(node):
        cardinality = graph.nodes[neighbor].get('cardinality', 0)
        graph.nodes[neighbor]['cardinality'] = cardinality + 1

    return node, degree


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

    # Create Var objects
    peo_var = [Var(var, size=graph.nodes[var]['size'],
                   name=graph.nodes[var]['name']) for var in peo]

    return peo_var, max_degree  # this is clique size - 1


@utils.sequential_profile_decorator(filename='fillin_graph_cprof')
def test_get_fillin_graph():
    """
    Test graph filling using the elimination order
    """
    import time
    import qtree.operators as ops
    nq, c = ops.read_circuit_file(
        'test_circuits/inst/cz_v2/10x10/inst_10x10_60_1.txt'
        # 'inst_2x2_7_1.txt'
    )
    g = circ2graph(nq, c, omit_terminals=False)

    peo = np.random.permutation(g.nodes)

    tim1 = time.time()
    g1 = get_fillin_graph(g, list(peo))
    tim2 = time.time()
    g2 = get_fillin_graph2(g, list(peo))
    tim3 = time.time()

    assert nx.is_isomorphic(g1, g2)
    print(tim2 - tim1, tim3 - tim2)


def test_is_zero_fillin():
    """
    Test graph filling using the elimination order
    """
    import time
    import qtree.operators as ops
    nq, c = ops.read_circuit_file(
        'test_circuits/inst/cz_v2/10x10/inst_10x10_60_1.txt'
    )
    g = circ2graph(nq, c, omit_terminals=False)

    g1 = get_fillin_graph(g, list(range(g.number_of_nodes())))

    tim1 = time.time()
    print(
        is_peo_zero_fillin(g1, list(range(g.number_of_nodes()))))
    tim2 = time.time()
    print(
        is_peo_zero_fillin2(g1, list(range(g.number_of_nodes()))))
    tim3 = time.time()

    print(tim2 - tim1, tim3 - tim2)


def test_maximum_cardinality_search():
    """Test maximum cardinality search algorithm"""

    # Read graph
    import qtree.operators as ops
    nq, c = ops.read_circuit_file(
        'inst_2x2_7_0.txt'
    )
    old_g = circ2graph(nq, c)

    # Make random clique
    vertices = list(np.random.choice(old_g.nodes, 4, replace=False))
    while is_clique(old_g, vertices):
        vertices = list(np.random.choice(old_g.nodes, 4, replace=False))

    g = make_clique_on(old_g, vertices)

    # Make graph completion
    peo, tw = get_peo(g)
    g_chordal = get_fillin_graph2(g, peo)

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
    import qtree.operators as ops
    nq, c = ops.read_circuit_file(
        'inst_2x2_7_0.txt'
    )
    g = circ2graph(nq, c)

    # select some random vertices
    vertices = list(np.random.choice(g.nodes, 4, replace=False))
    while is_clique(g, vertices):
        vertices = list(np.random.choice(g.nodes, 4, replace=False))

    g_new = make_clique_on(g, vertices)

    assert is_clique(g_new, vertices)


if __name__ == '__main__':
    test_get_fillin_graph()
    test_is_zero_fillin()
    test_maximum_cardinality_search()
    test_is_clique()


