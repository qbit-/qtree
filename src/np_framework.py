"""
This file implements Numpy framework of the
simulator. It's main use is in conjunction with the :py:mod:`optimizer`
module, and example programs are listed in :py:mod:`cirq_test` module.
"""

import numpy as np
import copy
import src.operators as ops
import src.optimizer as opt
import src.utils as utils


def get_np_buckets(buckets, data_dict, initial_state,
                   target_state, qubit_count):
    """
    Takes buckets and returns their Numpy counterparts.

    Parameters
    ----------
    buckets : list of list
              buckets as returned by :py:meth:`circ2buckets`
              and :py:meth:`reorder_buckets`.
    data_dict : dict
              dictionary containing values for the placeholder Tensors
    initial_state : int
              We estimate the amplitude of this initial_state state (ket).
    target_state : int
              We estimate the amplitude of target state (bra).
              Thus we know the values of circuit outputs
    qubit_count : int
              total number of qubits
    Returns
    -------
    np_buckets : list of lists
               Buckets having Numpy tensors in place of gate labels
    """
    # import pdb
    # pdb.set_trace()

    # Add input vectors

    # Create data for the input and output layers
    terminals_data_dict = {}

    for qubit_idx, qubit_value in enumerate(
            utils.qubit_vector_generator(initial_state, qubit_count)
    ):
        terminals_data_dict.update({
            (f'I{qubit_idx}', None): qubit_value
        })

    for qubit_idx, qubit_value in enumerate(
            utils.qubit_vector_generator(target_state, qubit_count)):
        terminals_data_dict.update({
            (f'O{qubit_idx}', None): qubit_value
        })

    # Create numpy buckets
    np_buckets = []
    for bucket in buckets:
        np_bucket = []
        for tensor in bucket:
            # sort tensor dimensions
            transpose_order = np.argsort(tensor.indices)
            try:
                data = data_dict[tensor.data_key]
            except KeyError:
                data = terminals_data_dict[tensor.data_key]

            new_tensor = opt.Tensor(tensor.name, tensor.indices,
                                    tensor.shape, data=np.transpose(
                                        data.copy(),
                                        transpose_order))
            new_tensor.transpose(transpose_order)

            np_bucket.append(new_tensor)
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
    tensor : optimizer.Tensor
           wrapper tensor object holding the result
    """
    result_indices = bucket[0].indices
    result_shape = bucket[0].shape
    result_data = bucket[0].data

    for tensor in bucket[1:]:
        expr = utils.get_einsum_expr(result_indices, tensor.indices)
        result_data = np.einsum(expr, result_data, tensor.data)

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
                        data=np.sum(result_data, axis=0))
    return result
