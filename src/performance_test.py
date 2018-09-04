"""
Performance testing for Qtree calculations
"""
import src.operators as ops
import src.optimizer as opt
import src.graph_model as gm
from src.logger_setup import log
import numpy as np
import time
from mpi4py import MPI
import tensorflow as tf
from src.cirq_test import extract_placeholder_dict
import pandas as pd
import subprocess
from matplotlib import pyplot as plt


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
    comput_graph = opt.bucket_elimination(
        tf_buckets, opt.process_bucket_tf)

    feed_dict = opt.assign_placeholder_values(
        placeholder_dict,
        target_state, n_qubits)
    amplitude = opt.run_tf_session(comput_graph, feed_dict)

    end_time = time.time()

    return end_time - start_time


def time_single_amplitude_numpy(
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

    np_buckets = opt.get_np_buckets(perm_buckets, n_qubits, target_state)
    amplitude = opt.bucket_elimination(
        np_buckets, opt.process_bucket_np)

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
            opt.bucket_elimination(
                sliced_tf_buckets, opt.process_bucket_tf),
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
        start_time = None

    # Synchronize processes
    env = comm.bcast(env, root=0)
    start_time = comm.bcast(start_time, root=0)

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

    end_time = time.time()
    elapsed_time = end_time - start_time

    comm.bcast(elapsed_time, root=0)

    return elapsed_time


def time_single_amplitude_mpi_numpy(
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

        env = dict(
            n_qubits=n_qubits,
            idx_parallel=idx_parallel,
            buckets=perm_buckets
        )
    else:
        env = None
        start_time = None

    # Synchronize processes
    env = comm.bcast(env, root=0)
    start_time = comm.bcast(start_time, root=0)

    # restore buckets
    buckets = env['buckets']

    # restore other parts of the environment
    n_qubits = env['n_qubits']
    idx_parallel = env['idx_parallel']

    # Transform label buckets to Numpy buckets
    np_buckets = opt.get_np_buckets(
        buckets, n_qubits, target_state)

    amplitude = 0
    for slice_dict in opt.slice_values_generator(
                comm_size, rank, idx_parallel):
        # Slice Numpy buckets along the parallelized vars
        sliced_buckets = opt.slice_np_buckets(
            np_buckets, slice_dict, idx_parallel)
        amplitude += opt.bucket_elimination(
            sliced_buckets, opt.process_bucket_np)

    amplitude = comm.reduce(amplitude, op=MPI.SUM, root=0)

    end_time = time.time()
    elapsed_time = end_time - start_time

    comm.bcast(elapsed_time, root=0)

    return elapsed_time


def collect_timings(
        out_filename,
        grid_sizes=[4, 5],
        depths=list(range(10, 15)),
        path_to_testcases='./test_circuits/inst/cz_v2',
        timing_fn=time_single_amplitude):
    """
    Runs timings for test circuits with grid size equal to grid_sizes
    and outputs results to a pandas.DataFrame, and saves to out_filename
    (as pickle).
    timing_fn is a sequential (non-MPI) function
    """

    try:
        data = pd.read_pickle(out_filename)
    except FileNotFoundError:
        # lays down the structure of data
        data = pd.DataFrame(
            [],
            index=['exec_time', 'total_time'],
            columns=pd.MultiIndex.from_product(
                [[], []],
                names=['grid size', 'depth']))

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
            exec_time = timing_fn(
                testfile, target_state)
            end_time = time.time()
            total_time = end_time - start_time

            # Merge current result with the rest
            data[grid_size, depth] = [exec_time, total_time]

    # Save result
    data.to_pickle(out_filename)

    return data


def collect_timings_mpi(
        out_filename,
        grid_sizes=[4, 5],
        depths=list(range(10, 15)),
        path_to_testcases='./test_circuits/inst/cz_v2',
        timing_fn_mpi=time_single_amplitude_mpi_numpy):
    """
    Runs timings for test circuits with grid size equal to grid_sizes
    and outputs results to a pandas.DataFrame, and saves to out_filename
    (as pickle).
    This version supports execution by mpiexec. Running
    mpiexec -n 1 python <:py:meth:`collect_timings_mpi`>
    will produce different timings than :py:meth:`collect_timings`
    timing_fn_mpi should be MPI-friendly timing function
    """

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank

    if rank == 0:
        try:
            data = pd.read_pickle(out_filename)
        except FileNotFoundError:
            # lays down the structure of data
            data = pd.DataFrame(
                [],
                index=['exec_time', 'total_time'],
                columns=pd.MultiIndex.from_product(
                    [[], []],
                    names=['grid size', 'depth']))

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
            n_var_parallel = int(np.floor(np.log2(comm_size)))

            # Synchronize processes
            comm.bcast(testfile, root=0)
            comm.bcast(target_state, root=0)

            # Measure time
            start_time = time.time()
            exec_time = timing_fn_mpi(
                testfile, target_state, n_var_parallel)
            end_time = time.time()
            total_time = end_time - start_time

            # Get maximal time as it determines overall time
            comm.reduce(exec_time, op=MPI.MAX, root=0)
            comm.reduce(total_time, op=MPI.MAX, root=0)

            if rank == 0:  # Parent process. Store results
                # Merge current result with the rest
                data[grid_size, depth] = [exec_time, total_time]

    if rank == 0:
        # Save result
        data.to_pickle(out_filename)

    return data


def collect_timings_for_multiple_processes(
        filename_base='output/test', n_processes=[1], extra_args=[]):
    """
    Run :py:meth:`collect_timings_mpi` with different number of mpi processes

    Parameters
    ----------
    filename_base : str
           base of the output filename to be appended with process #
    n_processes : list, default [1]
           number of processes
    extra_args : list, default []
           additional arguments to :py:meth:`collect_timings_mpi`
    """
    for n_proc in n_processes:
        filename = filename_base + '_' + str(n_proc) + '.p'
        sh = "mpiexec -n {} ".format(n_proc)
        sh += "python -c 'from src.performance_test import collect_timings_mpi;collect_timings_mpi(\"{}\",{})'".format(
            filename, ','.join(map(str, extra_args)))
        print(sh)

        process = subprocess.Popen(sh, shell=True)
        process.communicate()


def extract_parallel_efficiency(
        seq_filename, par_filename_base,
        n_processes=[1, 2], grid_size=4,
        depth=10, time_id='exec_time'):
    """
    Calculates parallel efficiency from collected data
    """
    seq_data = pd.read_pickle(seq_filename)
    seq_time = seq_data[(grid_size, depth)][time_id]

    par_times = []
    efficiencies = []
    for n_proc in n_processes:
        filename = par_filename_base + '_' + str(n_proc) + '.p'
        par_data = pd.read_pickle(filename)
        par_time = par_data[(grid_size, depth)][time_id]

        par_times.append(par_time)
        efficiencies.append(seq_time / (par_time * n_proc))

    return efficiencies, n_processes


def extract_timings_vs_gridsize(
        filename, grid_sizes,
        depth=10, time_id='exec_time'):
    """
    Extracts timings vs grid size from the timings data file
    for a fixed depth
    """
    data = pd.read_pickle(filename)

    times = []
    for grid_size in grid_sizes:
        time = data[(grid_size, depth)][time_id]
        times.append(time)

    return times, grid_sizes


def extract_timings_vs_depth(
        filename, depths,
        grid_size=4, time_id='exec_time'):
    """
    Extracts timings vs depth from the timings data file
    for a fixed grid_size
    """
    data = pd.read_pickle(filename)

    times = []
    for depth in depths:
        time = data[(grid_size, depth)][time_id]
        times.append(time)

    return times, depths


def plot_time_vs_depth(filename,
                       fig_filename='time_vs_depth.png',
                       interactive=False):
    """
    Plots time vs depth for some number of grid sizes
    Data is loaded from filename
    """
    if not interactive:
        plt.switch_backend('agg')

    grid_sizes = [4, 5]
    depths = range(10, 20)

    # Create empty canvas
    fig, axes = plt.subplots(1, len(grid_sizes), sharey=True,
                             figsize=(12, 6))

    for n, grid_size in enumerate(grid_sizes):
        time, depths = extract_timings_vs_depth(
            filename, depths, grid_size)
        axes[n].semilogy(depths, time)
        axes[n].set_xlabel(
            'depth of {}x{} circuit'.format(grid_size, grid_size))
        axes[n].set_ylabel('log(time in seconds)')
    fig.suptitle('Evaluation time vs depth of the circuit')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_par_vs_depth_multiple(
        seq_filename, par_filename_base,
        n_processes=[1, 2], fig_filename='time_vs_depth_multiple.png',
        interactive=False):
    """
    Plots time vs depth for sequential and multiple MPI
    runs
    """
    grid_size = 5
    depths = list(range(10, 21))

    if not interactive:
        plt.switch_backend('agg')

    # Create empty canvas
    fig, axes = plt.subplots(1, len(n_processes)+1, sharey=True,
                             figsize=(12, 6))

    filenames = [seq_filename] + [par_filename_base + '_' +
                                  str(n_proc) + '.p'
                                  for n_proc in n_processes]
    titles = ['Sequential'] + ['n = {}'.format(n_proc)
                               for n_proc in n_processes]

    for n, (filename, title) in enumerate(zip(filenames, titles)):
        time, depths = extract_timings_vs_depth(
            filename, depths, grid_size)
        axes[n].semilogy(depths, time)
        axes[n].set_xlabel('depth')
        axes[n].set_ylabel('log(time in seconds)')
        axes[n].set_title(title)

    fig.suptitle('Evaluation time vs depth of the circuit')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_par_efficiency(
        seq_filename, par_filename_base,
        n_processes=[1, 2], fig_filename='efficiency.png',
        interactive=False):
    """
    Plots parallel efficiency for a given set of processors
    """
    grid_size = 4
    depth = 10

    if not interactive:
        plt.switch_backend('agg')

    efficiency, n_proc = extract_parallel_efficiency(
        seq_filename, par_filename_base,
        n_processes, grid_size, depth,
    )

    # Create empty canvas
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(n_proc, efficiency)
    ax.set_xlabel(
            'number of processes')
    ax.set_ylabel('Efficiency')
    ax.set_title('Efficiency of MPI parallel code')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


if __name__ == "__main__":
    # collect_timings('output/test_numpy.p', [4, 5], list(range(10, 21)),
    #                 timing_fn=time_single_amplitude_numpy)
    collect_timings_for_multiple_processes(
        'output/test_numpy', [1, 2, 4, 8],
        extra_args=[[4, 5], list(range(10, 21))]
    )
    # plot_time_vs_depth('output/test.p', interactive=True)
    # plot_par_vs_depth_multiple('output/test.p', 'output/test', [1, 2, 4, 8], interactive=True)
