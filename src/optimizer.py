"""
Operations to load/contract quantum circuits
"""
import networkx as nx
import numpy as np
import tensorflow as tf
from src.logger_setup import log
import src.operators as ops

UP = np.array([1, 0])
DOWN = np.array([0, 1])


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


def get_tf_buckets(buckets, qubit_count):
    """
    Takes buckets and returns their Tensorflow counterparts, along
    with a placeholder dictionary to fill with actual gate values
    later.

    Parameters
    ----------
    buckets : list of list
              buckets as returned by :py:meth:`circ2buckets`
              and :py:meth:`transform_buckets`.
    qubit_count : int
              total number of qubits

    Returns
    -------
    tf_buckets : list of lists
               Buckets having Tensorflow tensors in place of gate labels
    placeholder_dict : dict
               Dictionary containing {gate_label : tensorflow_placeholder}
               pairs
    """
    # import pdb
    # pdb.set_trace()

    # Define ingredients
    X_1_2 = tf.placeholder(tf.complex64, [2, 2], 'X_1_2')
    Y_1_2 = tf.placeholder(tf.complex64, [2, 2], 'Y_1_2')
    H = tf.placeholder(tf.complex64, [2, 2], 'H')
    cZ = tf.placeholder(tf.complex64, [2, 2], 'cZ')
    T = tf.placeholder(tf.complex64, [2], 'T')

    placeholder_dict = {'X_1_2': X_1_2, 'Y_1_2': Y_1_2,
                        'H': H, 'cZ': cZ, 'T': T}

    # Add input vectors
    input_names = ['I{}'.format(ii)
                   for ii in range(1, qubit_count+1)]
    inputs = [tf.placeholder(tf.complex64, [2], name)
              for name in input_names]

    placeholder_dict.update(
        dict(zip(
            input_names, inputs))
    )

    # Add output vectors
    output_names = ['O{}'.format(ii)
                    for ii in range(1, qubit_count+1)]
    outputs = [tf.placeholder(tf.complex64, [2], name)
               for name in output_names]

    placeholder_dict.update(
        dict(zip(
            output_names, outputs))
    )

    # Create tf buckets from unordered buckets
    tf_buckets = []
    for bucket in buckets:
        tf_bucket = []
        for label, variables in bucket:

            # sort tensor dimensions (reversed order)
            transpose_order = np.argsort(variables)
            variables = sorted(variables)

            tf_bucket.append(
                (
                    tf.transpose(placeholder_dict[label],
                                 perm=transpose_order),
                    variables
                )
            )
        tf_buckets.append(tf_bucket)

    return tf_buckets, placeholder_dict


def int_to_bitstring(integer, width):
    """
    Transforms an integer to its bitsting of a specified width

    Parameters
    ----------
    integer : int
           integer number
    width : int
           width of the binary representation (padded with zeros)

    Returns
    -------
    bitsting : str
           string of binary representation of integer
    """
    bitstring = bin(integer)[2:]
    if len(bitstring) < width:
        bitstring = '0' * (width - len(bitstring)) + bitstring
    return bitstring


def qubit_vector_generator(target_state, n_qubits):
    """
    Generates a sequence of qubits corresponding to the
    binary representation of the target_state.

    Parameters
    ----------
    target_state : int
            integer whose binary representation encodes the target_state
    n_qubits : int
            number of qubits the target state describes

    Yields
    ------
    qubit : numpy.array of size [2]
          UP or DOWN state of a single qubit
    """
    bitstring = int_to_bitstring(target_state, n_qubits)
    for bit in bitstring:
        yield DOWN if bit == '0' else UP


def assign_placeholder_values(placeholder_dict, target_state, n_qubits):
    """
    Builds feed dictionary for Tensorflow from the placeholder dictionary,    which holds placeholders of all gates in the circuit,
    and target state.

    Parameters
    ----------
    placeholder_dict : dict
           Dictionary of {label : tensorflow_placeholder} pairs
    target_state : int
           Integer which encodes the state of qubits
    n_qubits : int
           Number of qubits in the circuit

    Returns
    -------
    feed_dict : dict
          Dictionary to feed in Tensorflow session
    """
    # Actual values of gates
    values_dict = ops.operator_matrices_dict

    # Fill all common gate's placeholders
    feed_dict = {placeholder_dict[key]: values_dict[key]
                 for key in values_dict.keys()}

    # Fill placeholders for the input layer
    input_dict = {}
    for ii, bc_state in enumerate(
            qubit_vector_generator(0, n_qubits)):
        input_dict.update({
            # numeration starts at 1!
            'I{}'.format(ii+1): bc_state @ values_dict['H']
        })
    input_feed_dict = {placeholder_dict[key]                       : val for key, val in input_dict.items()}

    feed_dict.update(input_feed_dict)

    # fill placeholders for the output layer
    output_dict = {}
    for ii, bc_state in enumerate(
            qubit_vector_generator(target_state, n_qubits)):
        output_dict.update({
            # numeration starts at 1!
            'O{}'.format(ii+1): values_dict['H'] @ bc_state
        })
    output_feed_dict = {placeholder_dict[key]                        : val for key, val in output_dict.items()}

    feed_dict.update(output_feed_dict)

    return feed_dict


def slice_tf_buckets(tf_buckets, old_pdict, idx_parallel):
    """
    Takes (symbolic) slices of the Tensorflow buckets
    over the variables in idx_parallel. Updates the placeholder
    dictionary.

    Parameters
    ----------
    tf_buckets : list of lists
              Buckets containing Tensorflow tensors and variables
    old_pdict : dict
              Placeholder dictionary
    idx_parallel : list
              Indices to parallelize over

    Returns
    -------
    sliced_buckets : list of lists
              buckets with (symbolically) sliced gates
    pdict : dict
              updated placeholder dictionary
    """
    # import pdb
    # pdb.set_trace()

    pdict = {key: val for key, val in old_pdict.items()}
    # Define slice variables
    slice_var_dict = {'q_{}'.format(var):
                      tf.placeholder(dtype=tf.int32,
                                     shape=[], name='q_{}'.format(var))
                      for var in idx_parallel}
    pdict.update(slice_var_dict)

    # Create tf buckets from unordered buckets
    sliced_buckets = []
    for bucket in tf_buckets:
        sliced_bucket = []
        for tensor, variables in bucket:
            slice_bounds = []
            new_shape = []
            for var in variables:
                if var in idx_parallel:
                    slice_bounds.append(slice_var_dict[f'q_{var}'])
                    new_shape.append(1)
                else:
                    slice_bounds.append(slice(None))
                    new_shape.append(2)
            sliced_bucket.append(
                (tf.reshape(tensor[slice_bounds], new_shape), variables)
            )
        sliced_buckets.append(sliced_bucket)

    return sliced_buckets, pdict


def slice_values_generator(comm_size, rank, idx_parallel):
    """
    Generates dictionaries containing consequtive values for
    each variable we parallelized over.

    Parameters
    ----------
    comm_size : int
            number of parallel workers
    rank : int
            parallel worker identificator
    idx_parallel : list
            variables to parallelize over

    Yields
    ------
    slice_dict : dict
            dictionary of {idx_parallel : value} pairs
    """
    var_names = ["q_{}".format(var) for var in idx_parallel]

    # iterate over all possible values of variables idx_parallel
    for ii in range(rank, 2**len(idx_parallel), comm_size):
        bitstring = int_to_bitstring(integer=ii, width=len(idx_parallel))
        yield dict(zip(var_names, bitstring))


def run_tf_session(tf_variable, feed_dict):
    """
    Run Tensorflow session and get variable value

    Parameters
    ----------
    tf_variable : tensorflow.Tensor
               variable to evaluate
    feed_dict : dict
               dictionary with placeholder values
    Returns
    -------
    res : numpy.array
               result of the calculation
    """
    with tf.Session() as sess:
        res = sess.run(tf_variable, feed_dict=feed_dict)

    return res


def num_to_alpha(integer):
    """
    Transform integer to [a-z], [a0-z0]-[a9-z9]

    Parameters
    ----------
    integer : int
        Integer to transform

    Returns
    -------
    a : str
        alpha-numeric representation of the integer
    """
    ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    if integer < 26:
        return ascii_lowercase[integer]
    elif integer < 261:
        return ascii_lowercase[integer % 25 - 1] + str(integer // 25)
    else:
        raise ValueError('Too large index for einsum')


def get_einsum_expr(idx1, idx2):
    """
    Takes two tuples of indice and returns an einsum expression
    to evaluate the sum over repeating indices

    Parameters
    ----------
    idx1 : list-like
          indices of the first argument
    idx2 : list-like
          indices of the second argument

    Returns
    -------
    expr : str
          Einsum command to sum over indices repeating in idx1
          and idx2.
    """
    result_indices = sorted(list(set(idx1 + idx2)))
    # remap indices to reduce their order, as einsum does not like
    # large numbers
    idx_to_least_idx = {old_idx: new_idx for new_idx, old_idx
                        in enumerate(result_indices)}

    str1 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in idx1)
    str2 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in idx2)
    str3 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in result_indices)
    return str1 + ',' + str2 + '->' + str3


def process_bucket(bucket):
    """
    Process bucket in the bucket elimination algorithm.
    We multiply all tensors in the bucket and sum over the
    variable which the bucket corresponds to. This way the
    variable of the bucket is removed from the expression.

    Parameters
    ----------
    bucket : list
           List containing tuples of tensors (gates) with their indices.

    Returns
    -------
    tensor : tuple
           array and a list of its indices
    """
    result, variables = bucket[0]

    for tensor, variables_current in bucket[1:]:
        expr = get_einsum_expr(variables, variables_current)
        result = tf.einsum(expr, result, tensor)
        variables = sorted(list(set(variables + variables_current)))

    if len(variables) > 1:
        variables = variables[1:]
    else:
        variables = []

    # reduce
    return tf.reduce_sum(result, axis=0), variables


def bucket_elimination(buckets):
    """
    Algorithm to evaluate a contraction of a large number of tensors.
    The variables to contract over are assigned ``buckets`` which
    hold tensors having respective variables. The algorithm
    proceeds through contracting one variable at a time, thus we aliminate    buckets one by one.

    Parameters
    ----------
    buckets : list of lists

    Returns
    -------
    result : tensor (0 dimensional)
    """
    # import pdb
    # pdb.set_trace()
    result = None
    for n, bucket in enumerate(buckets):
        if len(bucket) > 0:
            tensor, variables = process_bucket(bucket)
            if len(variables) > 0:
                first_index = variables[0]
                buckets[first_index-1].append((tensor, variables))
            else:
                if result is not None:
                    result *= tensor
                else:
                    result = tensor
    return result
