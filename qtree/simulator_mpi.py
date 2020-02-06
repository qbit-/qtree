import numpy as np
from mpi4py import MPI

import qtree.utils as utils
import qtree.np_framework as npfr
import qtree.optimizer as opt
from qtree.simulator import get_amplitudes_from_cirq
from qtree.simulator import prepare_parallel_evaluation_np


def eval_circuit_np_parallel_mpi(filename, initial_state=0):
    """
    Evaluate quantum circuit using MPI to parallelize
    over some of the variables.
    """
    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank

    # number of variables to split by parallelization
    # this should be adjusted by the algorithm from memory/cpu
    # requirements
    n_var_parallel = 2
    if rank == 0:
        env = prepare_parallel_evaluation_np(filename, n_var_parallel)
    else:
        env = None

    env = comm.bcast(env, root=0)

    # restore other parts of the environment
    bra_vars = env['bra_vars']
    ket_vars = env['ket_vars']
    vars_parallel = env['vars_parallel']

    # restore buckets
    buckets = env['buckets']

    # restore data dictionary
    data_dict = env['data_dict']

    # Construct slice dictionary for initial state
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)

    # Loop over all amplitudes
    amplitudes = []
    for target_state in range(2**len(bra_vars)):
        # Construct slice dictionary for the target state
        slice_dict.update(
            utils.slice_from_bits(target_state, bra_vars))

        # main computation loop. Populate respective slices
        # and do contraction

        amplitude = 0
        for parallel_slice_dict in utils.slice_values_generator(
                vars_parallel, rank, comm_size):
            slice_dict.update(parallel_slice_dict)
            sliced_buckets = npfr.get_sliced_np_buckets(
                buckets, data_dict, slice_dict)
            result = opt.bucket_elimination(
                sliced_buckets, npfr.process_bucket_np)

            amplitude += result.data

        amplitude = comm.reduce(amplitude, op=MPI.SUM, root=0)
        amplitudes.append(amplitude)

    if rank == 0:
        amplitudes_reference = get_amplitudes_from_cirq(filename)
        print('Result:')
        print(np.round(np.array(amplitudes), 3))
        print('Reference:')
        print(np.round(amplitudes_reference, 3))
        print('Max difference:')
        print(np.max(np.array(amplitudes)
                     - np.array(amplitudes_reference)))

if __name__ == '__main__':
    eval_circuit_np_parallel_mpi('inst_2x2_7_0.txt')
