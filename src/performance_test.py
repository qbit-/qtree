"""
Performance testing for Qtree calculations
"""
import src.operators as ops
import src.optimizer as opt
import src.graph_model as gm
from src.logger_setup import log
import time
from mpi4py import MPI
import tensorflow as tf
from src.cirq_test import extract_placeholder_dict
import pandas as pd
import trace
import sys


def time_single_amplitude(
        filename, target_state,
        quickbb_command='./quickbb/run_quickbb_64.sh'):
    """
    Returns the time of a single amplitude evaluation.
    The circuit is loaded from the filename and the
    amplitude of the state is calculated. The time excludes
    file loading and quickbb operation.
    """
    n_qubits, circuit = ops.read_circuit_file(filename)

    # Convert circuit to buckets
    buckets, graph = opt.circ2buckets(circuit)

    # Calculate eleimination order with QuickBB
    peo, max_mem = gm.get_peo(graph)

    # Start measurement
    start_time = time.time()

    perm_buckets = opt.transform_buckets(buckets, peo)

    tf_buckets, placeholder_dict = opt.get_tf_buckets(perm_buckets, n_qubits)
    comput_graph = opt.bucket_elimination(tf_buckets)

    feed_dict = opt.assign_placeholder_values(
        placeholder_dict,
        target_state, n_qubits)
    amplitude = opt.run_tf_session(comput_graph, feed_dict)

    end_time = time.time()

    return end_time - start_time


def time_single_amplitude_mpi(
        filename, target_state, n_var_parallel=2,
        quickbb_command='./quickbb/run_quickbb_64.sh'):
    """
    Returns the time of a single amplitude evaluation.
    The circuit is loaded from the filename and the
    amplitude of the state is calculated. The time excludes
    file loading and quickbb operation.
    """

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank

    if rank == 0:
        # filename = 'inst_2x2_7_0.txt'
        n_qubits, circuit = ops.read_circuit_file(filename)

        # Prepare graphical model
        buckets, graph = opt.circ2buckets(circuit)

        # Run quickBB and get contraction order
        (peo, max_mem,
         idx_parallel, reduced_graph) = gm.get_peo_parallel_degree(
             graph, n_var_parallel)

        # Start time measurement
        start_time = time.time()

        # Permute buckets to the order of optimal contraction
        perm_buckets = opt.transform_buckets(
            buckets, peo + idx_parallel)

        # Transform tensor labels in buckets to tensorflow placeholders
        # Reset Tensorflow graph as it may store
        # all tensors ever used before
        tf.reset_default_graph()
        tf_buckets, placeholder_dict = opt.get_tf_buckets(
            perm_buckets, n_qubits)

        # Apply slicing as we parallelize over some variables
        sliced_tf_buckets, pdict_sliced = opt.slice_tf_buckets(
            tf_buckets, placeholder_dict, idx_parallel)

        # Do symbolic computation of the result
        result = tf.identity(
            opt.bucket_elimination(sliced_tf_buckets),
            name='result'
        )

        env = dict(
            n_qubits=n_qubits,
            idx_parallel=idx_parallel,
            input_names=list(pdict_sliced.keys()),
            tf_graph_def=tf.get_default_graph().as_graph_def()
        )
    else:
        env = None

    # Synchronize processes
    env = comm.bcast(env, root=0)

    # restore tensorflow graph, extract inputs and outputs
    tf.reset_default_graph()
    tf.import_graph_def(env['tf_graph_def'], name='')
    placeholder_dict = extract_placeholder_dict(
        tf.get_default_graph(),
        env['input_names']
    )
    result = tf.get_default_graph().get_tensor_by_name('result:0')

    # restore other parts of the environment
    n_qubits = env['n_qubits']
    idx_parallel = env['idx_parallel']

    feed_dict = opt.assign_placeholder_values(
        placeholder_dict,
        target_state, n_qubits)

    amplitude = 0
    for slice_dict in opt.slice_values_generator(
                comm_size, rank, idx_parallel):
        parallel_vars_feed = {
            placeholder_dict[key]: val for key, val
            in slice_dict.items()}

        feed_dict.update(parallel_vars_feed)
        amplitude += opt.run_tf_session(result, feed_dict)

    amplitude = comm.reduce(amplitude, op=MPI.SUM, root=0)

    if rank == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
    else:
        elapsed_time = None

    comm.bcast(elapsed_time, root=0)

    return elapsed_time


def collect_timings(
        out_filename,
        grid_sizes=[4, 5],
        depths=list(range(10, 15)),
        path_to_testcases='./test_circuits/inst/cz_v2'):
    """
    Runs timings for test circuits with grid size equal to grid_sizes
    and outputs results to a pandas.DataFrame, and saves to out_filename
    (as pickle).
    """

    try:
        data = pd.read_pickle(out_filename)
    except FileNotFoundError:
        data = pd.DataFrame()

    total_tests = len(grid_sizes)*len(depths)
    log.info(f'Will run {total_tests} tests')

    # suffix of the test case file. Should we make it random?
    test_id = 2
    for n_grid, grid_size in enumerate(grid_sizes):
        log.info('Running grid = {}, [{}/{}]'.format(
            grid_size, n_grid+1, len(grid_sizes)))

        for n_depth, depth in enumerate(depths):
            log.info('Running depth = {}, [{}/{}]'.format(
                depth, n_depth+1, len(depths)))

            testfile = '/'.join((
                path_to_testcases,
                f'{grid_size}x{grid_size}',
                f'inst_{grid_size}x{grid_size}_{depth}_{test_id}.txt'
            ))

            # Will calculate "1111...1" target amplitude
            target_state = 2**(grid_size**2) - 1

            # Measure time
            start_time = time.time()
            exec_time = time_single_amplitude(testfile, target_state)
            end_time = time.time()
            total_time = end_time - start_time

            # Turn result into a pandas.Dataframe for storing
            col_index = pd.MultiIndex.from_product(
                [[grid_size], [depth]],
                names=['grid size', 'depth'])

            res = pd.DataFrame([exec_time, total_time],
                               index=['exec_time', 'total_time'],
                               columns=col_index)

            # Merge current result with the rest
            data = pd.merge(data, res, left_index=True, right_index=True)

    # Save result
    data.to_pickle(out_filename)

    return data


def collect_timings_mpi(
        out_filename,
        grid_sizes=[4, 5],
        depths=list(range(10, 15)),
        path_to_testcases='./test_circuits/inst/cz_v2'):
    """
    Runs timings for test circuits with grid size equal to grid_sizes
    and outputs results to a pandas.DataFrame, and saves to out_filename
    (as pickle).
    This version supports execution by mpiexec. Running
    mpiexec -n 1 python <:py:meth:`collect_timings_mpi`>
    will produce different timings than :py:meth:`collect_timings`
    """

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank

    if rank == 0:
        try:
            data = pd.read_pickle(out_filename)
        except FileNotFoundError:
            data = pd.DataFrame()

        total_tests = len(grid_sizes)*len(depths)
        log.info(f'Will run {total_tests} tests')
        log.info(f'Will run {comm_size} paralell processes')
    else:
        data = None

    # suffix of the test case file. Should we make it random?
    test_id = 2
    for n_grid, grid_size in enumerate(grid_sizes):
        if rank == 0:
            log.info('Running grid = {}, [{}/{}]'.format(
                grid_size, n_grid+1, len(grid_sizes)))

        for n_depth, depth in enumerate(depths):
            if rank == 0:
                log.info('Running depth = {}, [{}/{}]'.format(
                    depth, n_depth+1, len(depths)))

            testfile = '/'.join((
                path_to_testcases,
                f'{grid_size}x{grid_size}',
                f'inst_{grid_size}x{grid_size}_{depth}_{test_id}.txt'
            ))

            # Will calculate "1111...1" target amplitude
            target_state = 2**(grid_size**2) - 1

            # Set the number of parallelized variables ~ number of threads

            # Synchronize processes
            comm.bcast(testfile, root=0)
            comm.bcast(target_state, root=0)

            # Measure time
            start_time = time.time()
            exec_time = time_single_amplitude_mpi(testfile, target_state)
            end_time = time.time()
            total_time = end_time - start_time

            if rank == 0:  # Parent process. Store results
                # Turn result into a pandas.Dataframe for storing
                col_index = pd.MultiIndex.from_product(
                    [[grid_size], [depth]],
                    names=['grid size', 'depth'])

                res = pd.DataFrame([exec_time, total_time],
                                   index=['exec_time', 'total_time'],
                                   columns=col_index)

                # Merge current result with the rest
                data = pd.merge(data, res,
                                left_index=True, right_index=True)

    if rank == 0:
        # Save result
        data.to_pickle(out_filename)

    return data


if __name__ == "__main__":
    collect_timings_mpi('test.p')
