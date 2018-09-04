"""
Operations to load/contract quantum circuits. All functions
operating on Buckets (without any specific framework) should
go here.
"""
import networkx as nx
from src.logger_setup import log
import src.operators as ops


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
    g : networkx.Graph
            contraction graph of the circuit
    """
    # import pdb
    # pdb.set_trace()
    g = nx.Graph()

    qubit_count = len(circuit[0])
    # print(qubit_count)

    # Let's build an undirected graph for variables
    # we start from 1 here to avoid problems with quickbb
    for i in range(1, qubit_count+1):
        g.add_node(i)

    # Build buckets for bucket elimination algorithm along the way.
    # we start from 1 here to follow the variable indices
    buckets = []
    for ii in range(1, qubit_count+1):
        buckets.append(
            [[f'O{ii}', [ii]]]
        )

    current_var = qubit_count
    layer_variables = list(range(1, qubit_count+1))

    for layer in reversed(circuit[1:-1]):
        for op in layer:
            if not op.diagonal:
                # Non-diagonal gate adds a new variable and
                # an edge to graph
                var1 = layer_variables[op._qubits[0]]
                var2 = current_var+1

                g.add_node(var2)
                g.add_edge(var1, var2)

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
                    var1, var2
                )

                # append cZ gate to the bucket of lower variable index
                var1, var2 = sorted([var1, var2])
                buckets[var1-1].append(
                    [op.name, [var1, var2]]
                )

            if isinstance(op, ops.T):
                var1 = layer_variables[op._qubits[0]]
                # Do not add any variables (buckets), but add tensor
                # to the bucket
                buckets[var1-1].append(
                    [op.name, [var1, ]]
                )

    # add last layer of measurement vectors
    for qubit_idx, var in zip(range(1, qubit_count+1),
                              layer_variables):
        buckets[var-1].append(
            [f'I{qubit_idx}', [var, ]]
        )

    v = g.number_of_nodes()
    e = g.number_of_edges()

    log.info(f"Generated graph with {v} nodes and {e} edges")
    log.info(f"last index contains from {layer_variables}")

    # with io.StringIO() as outstrings:
    #     aj = nx.adjacency_matrix(g)
    #     np.savetxt(outstrings, aj.toarray(), delimiter=" ",fmt='%i')
    #     s = outstrings.getvalue()
    #     log.info("Adjacency matrix:\n" + s.replace('0','-'))

    # plt.figure(figsize=(10,10))
    # nx.draw(g, with_labels=True)
    # plt.savefig('graph.eps')
    return buckets, g


def transform_buckets(old_buckets, permutation):
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


def bucket_elimination(buckets, process_bucket_fn):
    """
    Algorithm to evaluate a contraction of a large number of tensors.
    The variables to contract over are assigned ``buckets`` which
    hold tensors having respective variables. The algorithm
    proceeds through contracting one variable at a time, thus we aliminate    buckets one by one.

    Parameters
    ----------
    buckets : list of lists
    process_bucket_fn : function that will process this kind of buckets

    Returns
    -------
    result : tensor (0 dimensional)
    """
    # import pdb
    # pdb.set_trace()
    result = None
    for n, bucket in enumerate(buckets):
        if len(bucket) > 0:
            tensor, variables = process_bucket_fn(bucket)
            if len(variables) > 0:
                first_index = variables[0]
                buckets[first_index-1].append((tensor, variables))
            else:
                if result is not None:
                    result *= tensor
                else:
                    result = tensor
    return result
