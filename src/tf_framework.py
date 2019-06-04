"""
This file implements Tensorflow framework of the
simulator. It's main use is in conjunction with the :py:mod:`optimizer`
module, and example programs are listed in :py:mod:`cirq_test` module.
"""

import numpy as np
import tensorflow as tf
import src.operators as ops
import src.optimizer as opt
import src.utils as utils
import src.system_defs as defs


def get_tf_buckets(buckets):
    """
    Takes buckets and returns their Tensorflow counterparts, where
    all data attributes of tensors are filled with Tensorflow
    placeholders.

    Parameters
    ----------
    buckets : list of list
              buckets as returned by :py:meth:`read_buckets`
              and :py:meth:`reorder_buckets`.

    Returns
    -------
    tf_buckets : list of lists
               Buckets having Tensorflow tensors in place of Tensor.data
               attribute
    placeholder_dict: dict
               dictionary of the form {placeholder: data_key}.
               Varying tensors like I{qubit}, O{qubit} have their
               name in place of data_key
    """
    # import pdb
    # pdb.set_trace()

    placeholder_dict = {}
    # Create tf buckets from unordered buckets
    tf_buckets = []
    for bucket in buckets:
        tf_bucket = []
        for tensor in bucket:
            # sort tensor dimensions
            transpose_order = np.argsort(tensor.indices)
            placeholder = tf.placeholder(defs.TF_ARRAY_TYPE,
                                         tensor.shape, tensor.name)

            # Save the reference to placeholder in the dictionary
            placeholder_dict[placeholder] = tensor.data_key

            # Create new tensor with a placeholder for data
            new_tensor = opt.Tensor(tensor.name, tensor.indices,
                                    tensor.shape, data=tf.transpose(
                                        placeholder, transpose_order))
            new_tensor.transpose(transpose_order)
            tf_bucket.append(new_tensor)

        tf_buckets.append(tf_bucket)

    return tf_buckets, placeholder_dict


def assign_placeholder_values(placeholder_dict, data_dict,
                              initial_state, target_state, n_qubits):
    """
    Builds feed dictionary for Tensorflow from the placeholder
    dictionary, which holds placeholders of all gates in the circuit,
    and target state.

    Parameters
    ----------
    placeholder_dict : dict
           Dictionary of {tensorflow.placeholder : data_key} pairs
    data_dict : dict
           Dictionary of {data_key : np.array} pairs
    initial_state : int
           Integer which encodes the initial state of qubits (ket),
           big endian
    target_state : int
           Integer which encodes the final state of qubits (bra),
           big endian
    n_qubits : int
           Number of qubits in the circuit

    Returns
    -------
    feed_dict : dict
          Dictionary to feed in Tensorflow session
    """

    # Create data for the input and output layers
    terminals_data_dict = {}
    for qubit_idx, qubit_value in enumerate(
            utils.qubit_vector_generator(initial_state, n_qubits)):
        terminals_data_dict.update({
            (f'I{qubit_idx}', None): qubit_value
        })

    for qubit_idx, qubit_value in enumerate(
            utils.qubit_vector_generator(target_state, n_qubits)):
        terminals_data_dict.update({
            (f'O{qubit_idx}', None): qubit_value
        })

    # Fill all fixed gates placeholders
    feed_dict = {}
    for placeholder, data_key in placeholder_dict.items():
        try:
            feed_dict[placeholder] = data_dict[data_key]
        except KeyError:
            feed_dict[placeholder] = terminals_data_dict[data_key]

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
    tensor : optimizer.Tensor
           wrapper tensor object holding the resulting computational graph
    """
    result_data = bucket[0].data
    result_indices = bucket[0].indices
    result_shape = bucket[0].shape

    for tensor in bucket[1:]:
        expr = utils.get_einsum_expr(result_indices, tensor.indices)
        result_data = tf.einsum(expr, result_data, tensor.data)
        # Merge and sort indices and shapes
        result_indices = tuple(sorted(set(result_indices
                                          + tensor.indices)))
        shapes_dict = dict(zip(tensor.indices, tensor.shape))
        shapes_dict.update(dict(zip(result_indices, result_shape)))

        result_shape = tuple(shapes_dict[idx] for idx in result_indices)

    if len(result_indices) > 0:
        first_index, *result_indices = result_indices
        result_shape = result_shape[1:]
    else:
        first_index = 'f'
        result_indices = []
        result_shape = []

    # reduce
    result = opt.Tensor(f'E{first_index}', result_indices,
                        result_shape,
                        data=tf.reduce_sum(result_data, axis=0))
    return result


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
