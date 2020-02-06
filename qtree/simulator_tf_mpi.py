import numpy as np
import tensorflow as tf

import qtree.utils as utils
import qtree.tf_framework as tffr

from qtree.simulator_tf import prepare_parallel_evaluation_tf
from qtree.simulator import get_amplitudes_from_cirq
from mpi4py import MPI

def eval_circuit_tf_parallel_mpi(filename, initial_state=0):
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
        env = prepare_parallel_evaluation_tf(filename, n_var_parallel)
    else:
        env = None

    env = comm.bcast(env, root=0)

    # restore tensorflow graph, extract inputs and outputs
    tf.reset_default_graph()
    tf.import_graph_def(env['tf_graph_def'], name='')
    tgraph = tf.get_default_graph()
    result = tgraph.get_tensor_by_name('result:0')

    # restore placeholder and data dictionaries
    picklable_placeholders = env['picklable_placeholders']
    placeholder_dict = {tgraph.get_tensor_by_name(key): val
                        for key, val in picklable_placeholders.items()}

    data_dict = env['data_dict']

    # restore other parts of the environment
    bra_vars = env['bra_vars']
    ket_vars = env['ket_vars']
    vars_parallel = env['vars_parallel']

    # Construct slice dictionary for initial state
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)

    # Construct part of the feed dictionary
    feed_dict = tffr.assign_tensor_placeholders(
        placeholder_dict, data_dict)
    feed_dict.update(tffr.assign_variable_placeholders(
        placeholder_dict, slice_dict
    ))

    # Loop over all amplitudes
    amplitudes = []
    for target_state in range(2**len(bra_vars)):
        # Construct slice dictionary for the target state and
        # populate feed dictionary with proper values
        slice_dict = utils.slice_from_bits(target_state, bra_vars)
        feed_dict.update(tffr.assign_variable_placeholders(
            placeholder_dict, slice_dict))

        amplitude = 0
        for parallel_slice_dict in utils.slice_values_generator(
                vars_parallel, rank, comm_size):
            # Update feed dict with proper slices for parallelized
            # variables
            feed_dict.update(tffr.assign_variable_placeholders(
                placeholder_dict, parallel_slice_dict))

            amplitude += tffr.run_tf_session(result, feed_dict)

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
    eval_circuit_tf_parallel_mpi('inst_2x2_7_0.txt')
