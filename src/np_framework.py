"""
This file implements Numpy framework of the
simulator. It's main use is in conjunction with the :py:mod:`optimizer`
module, and example programs are listed in :py:mod:`cirq_test` module.
"""

import numpy as np
import copy
import src.operators as ops
import src.utils as utils


def get_np_buckets(buckets, qubit_count, target_state):
    """
    Takes buckets and returns their Numpy counterparts.

    Parameters
    ----------
    buckets : list of list
              buckets as returned by :py:meth:`read_buckets`
              and :py:meth:`transform_buckets`.
    qubit_count : int
              total number of qubits
    target_state : int
              We estimate the amplitude of target state.
              Thus we know the values of circuit outputs
    Returns
    -------
    np_buckets : list of lists
               Buckets having Numpy tensors in place of gate labels
    """
    # import pdb
    # pdb.set_trace()

    # Define ingredients
    matrices_dict = copy.deepcopy(
        ops.operator_matrices_dict)

    # Add input vectors

    input_layer_dict = {
        'I{}'.format(qubit_idx): qubit_value @ matrices_dict['h']
        for qubit_idx, qubit_value
        in zip(
            range(1, qubit_count+1),
            utils.qubit_vector_generator(0, qubit_count)
        )
    }
    matrices_dict.update(input_layer_dict)

    # Add output vectors

    output_layer_dict = {
        'O{}'.format(qubit_idx): matrices_dict['h'] @ qubit_value
        for qubit_idx, qubit_value
        in zip(
            range(1, qubit_count+1),
            utils.qubit_vector_generator(target_state, qubit_count)
        )
    }
    matrices_dict.update(output_layer_dict)

    # Create tf buckets from unordered buckets
    np_buckets = []
    for bucket in buckets:
        np_bucket = []
        for label, variables in bucket:

            # sort tensor dimensions (reversed order)
            transpose_order = np.argsort(variables)
            variables = sorted(variables)
            tensor = np.array(
                matrices_dict[label],
                copy=True)
            np_bucket.append(
                (
                    np.transpose(tensor, transpose_order),
                    variables
                )
            )
        np_buckets.append(np_bucket)

    return np_buckets


def slice_np_buckets(np_buckets, slice_var_dict, idx_parallel):
    """
    Takes slices of the tensors in Numpy buckets
    over the variables in idx_parallel.

    Parameters
    ----------
    np_buckets : list of lists
              Buckets containing Numpy tensors
    slice_var_dict : dict
              Current values of the sliced variables
    idx_parallel : list
              Indices to parallelize over

    Returns
    -------
    sliced_buckets : list of lists
              buckets with sliced gates
    """
    # import pdb
    # pdb.set_trace()

    # Create tf buckets from unordered buckets
    sliced_buckets = []
    for bucket in np_buckets:
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
                (np.reshape(
                    tensor[tuple(slice_bounds)],
                    new_shape),
                 variables)
            )
        sliced_buckets.append(sliced_bucket)

    return sliced_buckets


def process_bucket_np(bucket):
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
        result = np.einsum(expr, result, tensor)
        variables = sorted(list(set(variables + variables_current)))

    if len(variables) > 1:
        variables = variables[1:]
    else:
        variables = []

    # reduce
    return np.sum(result, axis=0), variables
