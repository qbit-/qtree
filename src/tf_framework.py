"""
This file implements Tensorflow framework of the
simulator. It's main use is in conjunction with the :py:mod:`optimizer`
module, and example programs are listed in :py:mod:`cirq_test` module.
"""

import numpy as np
import tensorflow as tf
import src.operators as ops
import src.utils as utils


def get_tf_buckets(buckets, qubit_count):
    """
    Takes buckets and returns their Tensorflow counterparts, along
    with a placeholder dictionary to fill with actual gate values
    later.

    Parameters
    ----------
    buckets : list of list
              buckets as returned by :py:meth:`read_buckets`
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
    X_1_2 = tf.placeholder(tf.complex64, [2, 2], 'x_1_2')
    Y_1_2 = tf.placeholder(tf.complex64, [2, 2], 'y_1_2')
    H = tf.placeholder(tf.complex64, [2, 2], 'h')
    cZ = tf.placeholder(tf.complex64, [2, 2], 'cz')
    T = tf.placeholder(tf.complex64, [2], 't')

    placeholder_dict = {'x_1_2': X_1_2, 'y_1_2': Y_1_2,
                        'h': H, 'cz': cZ, 't': T}

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
            utils.qubit_vector_generator(0, n_qubits)):
        input_dict.update({
            # numeration starts at 1!
            'I{}'.format(ii+1): bc_state @ values_dict['h']
        })
    input_feed_dict = {placeholder_dict[key]                       : val for key, val in input_dict.items()}

    feed_dict.update(input_feed_dict)

    # fill placeholders for the output layer
    output_dict = {}
    for ii, bc_state in enumerate(
            utils.qubit_vector_generator(target_state, n_qubits)):
        output_dict.update({
            # numeration starts at 1!
            'O{}'.format(ii+1): values_dict['h'] @ bc_state
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
                (tf.reshape(tensor[tuple(slice_bounds)], new_shape),
                 variables)
            )
        sliced_buckets.append(sliced_bucket)

    return sliced_buckets, pdict


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
    # Configure tensorflow for single threaded execution
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )

    with tf.Session(config=session_conf) as sess:
        res = sess.run(tf_variable, feed_dict=feed_dict)

    return res


def process_bucket_tf(bucket):
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
        expr = utils.get_einsum_expr(variables, variables_current)
        result = tf.einsum(expr, result, tensor)
        variables = sorted(list(set(variables + variables_current)))

    if len(variables) > 1:
        variables = variables[1:]
    else:
        variables = []

    # reduce
    return tf.reduce_sum(result, axis=0), variables


def extract_placeholder_dict(tf_graph, variable_names):
    """
    Extract placeholders from the tensorflow computation Graph.

    Returns
    -------
    pdict : dict
        List containing {label : tensorflow placeholder} pairs
    """
    return {
        name: tf_graph.get_tensor_by_name(name + ':0') for
        name in variable_names
    }
