"""
This module implements different utility functions
which don't definitely fit somewhere else. It also serves
for dependency disentanglement purposes.
"""
import numpy as np

ZERO = np.array([1, 0])
ONE = np.array([0, 1])


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
    The qubits are generated in the big-endian order

    Parameters
    ----------
    target_state : int
            integer whose binary representation encodes the target_state
    n_qubits : int
            number of qubits the target state describes

    Yields
    ------
    qubit : numpy.array of size [2]
          ZERO or ONE state of a single qubit
    """
    bitstring = int_to_bitstring(target_state, n_qubits)
    for bit in bitstring:
        yield ZERO if bit == '0' else ONE


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
        int_sequence = map(int, bitstring)
        yield dict(zip(var_names, int_sequence))


def num_to_alpha(integer):
    """
    Transform integer to [a-z], [A-Z]

    Parameters
    ----------
    integer : int
        Integer to transform

    Returns
    -------
    a : str
        alpha-numeric representation of the integer
    """
    ascii = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if integer < 52:
        return ascii[integer]
    else:
        raise ValueError('Too large index for einsum')


def num_to_alnum(integer):
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
    else:
        return ascii_lowercase[integer % 25 - 1] + str(integer // 25)


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
