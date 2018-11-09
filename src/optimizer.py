"""
Operations to load/contract quantum circuits. All functions
operating on Buckets (without any specific framework) should
go here.
"""

import itertools
import random
import re
import networkx as nx
import src.utils as utils
import src.operators as ops

from src.logger_setup import log

random.seed(0)


def read_buckets(filename, free_qubits=[], max_depth=None):
    """
    Reads circuit from filename and builds buckets for its contraction

    Parameters
    ----------
    filename : str
             circuit file in the format of Sergio Boixo
    free_qubits : list, optional
             qubits that have to be not contracted. Numeration
             is zero based
    max_depth : int
             maximal depth of gates to read

    Returns
    -------
    qubit_count : int
            number of qubits in the circuit
    buckets : list of lists
            list of lists (buckets)
    free_variables : list
            possible free variable list
    """
    # perform the cirquit file processing
    log.info(f'reading file {filename}')

    with open(filename, 'r') as fp:
        # read the number of qubits
        qubit_count = int(fp.readline())
        log.info("There are {:d} qubits in circuit".format(qubit_count))

        n_ignored_layers = 0
        current_layer = 0

        # Build buckets for bucket elimination algorithm.
        # we start with adding border 1-variable tensors
        buckets = []
        for ii in range(1, qubit_count+1):
            buckets.append(
                [[f'I{ii}', [ii]]]
            )

        layer_variables = list(range(1, qubit_count+1))
        current_var = qubit_count

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

                # append cZ gate to the bucket of lower variable index
                min_var = min(var1, var2)
                buckets[min_var-1].append(
                    [op_identif, [var1, var2]]
                )

            # Skip Hadamard tensors - for now
            elif op_identif == 'h':
                pass

            # Add selfloops for single variable gates
            elif op_identif == 't':
                var1 = layer_variables[q_idx[0]]
                # Do not add any variables (buckets), but add tensor
                # to the bucket
                buckets[var1-1].append(
                    [op_identif, [var1, ]]
                )

            # Process non-diagonal gates X and Y
            else:
                var1 = layer_variables[q_idx[0]]
                var2 = current_var+1

                # Append gate 2-variable tensor to the first variable's
                # bucket. This yields buckets containing variables
                # in increasing order (starting at least with bucket's
                # variable)
                buckets[var1-1].append(
                    [op_identif, [var1, var2]]
                )

                # Create a new variable
                buckets.append(
                    []
                )

                current_var += 1
                layer_variables[q_idx[0]] = current_var

        # add border tensors for the last layer
        for qubit_idx, var in zip(range(qubit_count),
                                  layer_variables):
            if qubit_idx not in free_qubits:
                buckets[var-1].append(
                    ['O{}'.format(qubit_idx+1), [var, ]]
                )

        # Now add Hadamards for free qubits
        for qubit_idx in free_qubits:
            var1 = layer_variables[qubit_idx]
            var2 = current_var+1

            # Append H gate
            buckets[var1-1].append(
                ['h', [var1, var2]]
            )

            # Create a new variable
            buckets.append(
                []
            )
            current_var += 1
            layer_variables[qubit_idx] = current_var

        # Collect free variables
        free_variables = [layer_variables[qubit_idx]
                          for qubit_idx in free_qubits]

        # We are done, print stats
        if n_ignored_layers > 0:
            log.info("Ignored {} layers".format(n_ignored_layers))

        n_variables = len(buckets)
        n_tensors = 0
        for bucket in buckets:
            n_tensors += len(bucket)

        log.info(
            f"Generated buckets with {n_variables} variables" +
            f" and {n_tensors} tensors")
        log.info(f"last index contains from {layer_variables}")

    return qubit_count, buckets, free_variables


def circ2buckets(circuit):
    """
    Takes circuit in the form of list of gate lists, builds
    its contraction graph and variable buckets. Buckets contain tuples
    corresponding to quantum gates and qubits they act on. Each bucket
    corresponds to a variable. Each bucket can hold gates acting on it's
    variable of variables with higher index.

    Parameters
    ----------
    circuit : list of lists
            quantum circuit as returned by
            :py:meth:`operators.read_circuit_file`

    Returns
    -------
    buckets : list of lists
            list of lists (buckets)
    g : networkx.MultiGraph
            contraction graph of the circuit
    """
    # import pdb
    # pdb.set_trace()
    g = nx.MultiGraph()

    qubit_count = len(circuit[0])

    # Let's build an undirected graph for variables
    # we start from 1 here to avoid problems with quickbb
    for var in range(1, qubit_count+1):
        g.add_node(ii, name=utils.num_to_alnum(ii))

    # Add selfloops to the border nodes
    for var in range(1, qubit_count+1):
        g.add_edge(
            var, var,
            tensor=f'I{var}',
            hash_tag=hash(
                (f'I{var}', (var, var),
                 random.random()))
        )

    # Build buckets for bucket elimination algorithm along the way.
    # we start from 1 here to follow the variable indices
    buckets = []
    for ii in range(1, qubit_count+1):
        buckets.append(
            [[f'I{ii}', [ii]]]
        )

    current_var = qubit_count
    layer_variables = list(range(1, qubit_count+1))

    for layer in circuit[1:-1]:
        for op in layer:
            if not op.diagonal:
                # Non-diagonal gate adds a new variable and
                # an edge to graph
                var1 = layer_variables[op._qubits[0]]
                var2 = current_var+1

                g.add_node(var2, name=utils.num_to_alnum(var2))
                g.add_edge(var1, var2,
                           tensor=op.name,
                           hash_tag=hash((
                               op.name, (var1, var2),
                               random.random()))
                )

                # Append gate 2-variable tensor to the first variable's
                # bucket. This yields buckets containing variables
                # in increasing order (starting at least with bucket's
                # variable)
                buckets[var1-1].append(
                    [op.name, [var1, var2]]
                )

                # Create a new variable
                buckets.append(
                    []
                )

                current_var += 1
                layer_variables[op._qubits[0]] = current_var

            if isinstance(op, ops.cZ):
                var1 = layer_variables[op._qubits[0]]
                var2 = layer_variables[op._qubits[1]]

                # cZ connects two variables with an edge
                g.add_edge(
                    var1, var2,
                    tensor=op.name,
                    hash_tag=hash(
                        (op.name, (var1, var2),
                         random.random()))
                )

                # append cZ gate to the bucket of lower variable index
                min_var = min(var1, var2)
                buckets[min_var-1].append(
                    [op.name, [var1, var2]]
                )

            if isinstance(op, ops.T):
                var1 = layer_variables[op._qubits[0]]
                # Do not add any variables (buckets), but add tensor
                # to the bucket
                buckets[var1-1].append(
                    [op.name, [var1, ]]
                )
                # Add a selfloop for a 1-variable tensor
                g.add_edge(
                    var1, var1,
                    tensor=op.name,
                    hash_tag=hash(
                        (op.name, (var1, var1),
                         random.random()))
                )

    # add first layer
    for qubit_idx, var in zip(range(1, qubit_count+1),
                              layer_variables):
        buckets[var-1].append(
            [f'O{qubit_idx}', [var, ]]
        )
    # add selfloops to the border edges
    for qubit_idx, var in zip(range(1, qubit_count+1),
                              layer_variables):
        g.add_edge(var, var,
                   tensor=f'O{qubit_idx}',
                   hash_tag=hash(
                       (f'O{qubit_idx}',
                        (var, var), random.random()))
        )

    v = g.number_of_nodes()
    e = g.number_of_edges()

    log.info(f"Generated graph with {v} nodes and {e} edges")
    log.info(f"last index contains from {layer_variables}")

    return buckets, g


def bucket_elimination(buckets, process_bucket_fn, n_var_nosum=0):
    """
    Algorithm to evaluate a contraction of a large number of tensors.
    The variables to contract over are assigned ``buckets`` which
    hold tensors having respective variables. The algorithm
    proceeds through contracting one variable at a time, thus we eliminate
    buckets one by one.

    Parameters
    ----------
    buckets : list of lists
    process_bucket_fn : function
              function that will process this kind of buckets
    n_var_nosum : int, optional
              number of variables that have to be left in the
              result. Expected at the end of bucket list
    Returns
    -------
    result : numpy.array
    """
    # import pdb
    # pdb.set_trace()
    n_var_contract = len(buckets) - n_var_nosum

    result = None
    for n, bucket in enumerate(buckets[:n_var_contract]):
        if len(bucket) > 0:
            tensor, variables = process_bucket_fn(bucket)
            if len(variables) > 0:
                first_index = variables[0]
                buckets[first_index-1].append((tensor, variables))
            else:   # tensor is scalar
                if result is not None:
                    result *= tensor
                else:
                    result = tensor

    rest = list(itertools.chain.from_iterable(buckets[n_var_contract:]))
    if len(rest) > 0:
        tensor, variables = process_bucket_fn(rest, nosum=True)
        if result is not None:
            result *= tensor
        else:
            result = tensor
    return result


def buckets2graph(buckets, ignore_variables=[]):
    """
    Takes buckets and produces a corresponding undirected graph. Single
    variable tensors are coded as self loops and there may be
    multiple parallel edges.

    !!!!!!!!!
    Warning! Conversion of buckets to graphs destroys the information
    about permutations of the input tensors. The restored buckets
    may not evaluate to the same result as the original ones.
    !!!!!!!!!

    Parameters
    ----------
    buckets : list of lists
    ignore_variables : list, optional
       Variables to be deleted from the resulting graph.
       Numbering is 1-based.

    Returns
    -------
    graph : networkx.MultiGraph
            contraction graph of the circuit
    """
    graph = nx.MultiGraph()

    # Let's build an undirected graph for variables
    for n, bucket in enumerate(buckets):
        for element in bucket:
            tensor, variables = element
            for var in variables:
                graph.add_node(var, name=utils.num_to_alnum(var))
            if len(variables) > 1:
                edges = itertools.combinations(variables, 2)
            else:
                # If this is a single variable tensor, add self loop
                var = variables[0]
                edges = [[var, var]]
            graph.add_edges_from(
                edges, tensor=tensor,
                hash_tag=hash(
                    (tensor, tuple(variables), random.random())
                )
            )

    # Delete any requested variables from the final graph
    if len(ignore_variables) > 0:
        graph.remove_nodes_from(ignore_variables)

    return graph


def graph2buckets(graph):
    """
    Takes a Networkx MultiGraph and produces a corresponding
    bucket list. This is an inverse of the :py:meth:`buckets2graph`

    !!!!!!!!!
    Warning! Conversion of buckets to graphs destroys the information
    about permutations of the input tensors. The restored buckets
    may not evaluate to the same result as the original ones.
    !!!!!!!!!

    Parameters
    ----------
    graph : networkx.MultiGraph
            contraction graph of the circuit. Has to support self loops
            and parallel edges. Parallel edges are needed to support
            multiple qubit operators on same qubits
            (which can be collapsed in one operation)

    Returns
    -------
    buckets : list of lists
    """
    buckets = []
    variables = sorted(graph.nodes)

    # Add buckets with sorted variables
    for n, variable in enumerate(variables):
        tensors_of_variable = list(graph.edges(variable, data=True))

        # First collect all unique tensors (they may be elements of
        # the current bucket)

        candidate_elements = {}

        # go over pairs of variables.
        # The first variable in pair is this variable
        for edge in tensors_of_variable:
            _, other_variable, edge_data = edge
            hash_tag = edge_data['hash_tag']
            tensor = edge_data['tensor']

            element = candidate_elements.get(hash_tag, None)
            if element is None:
                element = [tensor, [variable, other_variable]]
            else:
                element[1].append(other_variable)
            candidate_elements[hash_tag] = element

        # Now we have all unique tensors in bucket format.
        # Drop tensors where current variable is not the lowest in order

        bucket = []
        for element in candidate_elements.values():
            tensor, variables = element
            # Sort variables and also remove self loops used for single
            # variable tensors
            sorted_variables = sorted(set(variables))
            if sorted_variables[0] == variable:
                bucket.append([tensor, sorted_variables])
        buckets.append(bucket)

    return buckets


def reorder_buckets(old_buckets, permutation):
    """
    Transforms bucket list according to the new order given by
    permutation. The variables are renamed and buckets are reordered
    to hold only gates acting on variables with strongly increasing
    index.

    Parameters
    ----------
    old_buckets : list of lists
          old buckets
    permutation : list
          permutation of variables

    Returns
    -------
    new_buckets : list of lists
          buckets reordered according to permutation
    """
    # import pdb
    # pdb.set_trace()
    perm_table = dict(zip(permutation, range(1, len(permutation) + 1)))
    n_variables = len(old_buckets)
    new_buckets = []
    for ii in range(n_variables):
        new_buckets.append([])

    for bucket in old_buckets:
        for gate in bucket:
            label, variables = gate
            new_variables = [perm_table[ii] for ii in variables]
            bucket_idx = sorted(new_variables)[0]
            # we leave the variables permuted, as the permutation
            # will be needed to transform tensorflow tensor
            new_buckets[bucket_idx-1].append([label, new_variables])

    return new_buckets


def test_bucket_graph_conversion(filename):
    """
    Test the conversion between Buckets and the contraction multigraph
    """
    # load circuit
    n_qubits, buckets, free_vars = read_buckets(filename)
    graph = buckets2graph(buckets)
    buckets_new = graph2buckets(graph)
    graph_new = buckets2graph(buckets_new)

    buckets_equal = True
    for b1, b2 in zip(buckets, buckets_new):
        if sorted(b1) != sorted(b2):
            buckets_equal = False
            break

    print('Buckets equal? : {}'.format(buckets_equal))
    print('Graphs equal? : {}'.format(nx.is_isomorphic(graph, graph_new)))


if __name__ == '__main__':
    test_bucket_graph_conversion('inst_2x2_7_0.txt')
