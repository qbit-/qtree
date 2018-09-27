"""
This module is for the experimentation with splitting
graphs in different ways to reduce the computational effort
of the tensor contraction
"""
import numpy as np
import pandas as pd
import itertools

import src.operators as ops
import src.optimizer as opt
import src.graph_model as gm

from src.logger_setup import log
from matplotlib import pyplot as plt


def get_cost_vs_parallel_size(filename, step_by=1, start_at=0, stop_at=None):
    """
    Calculates memory cost vs the number of parallelized
    variables for a given circuit.

    Parameters
    ----------
    filename : str
           input file
    Returns
    -------
          max_mem - maximal memory (if all intermediates are kept)
          min_mem - minimal possible memory for the algorithm
          total_max_mem - maximal memory of all tasks combined
          total_min_mem - minimal memory of all tasks combined
          treewidth - treewidth returned by quickBB
    """
    n_qubits, circuit = ops.read_circuit_file(filename)

    buckets, _ = opt.circ2buckets(circuit)
    graph_raw = opt.buckets2graph(buckets)
    n_var_total = graph_raw.number_of_nodes()
    if stop_at is None or stop_at > n_var_total:
        stop_at = n_var_total

    results = []
    for n_var_parallel in range(start_at, stop_at, step_by):
        idx_parallel, reduced_graph = gm.split_graph_by_metric(
             graph_raw, n_var_parallel)

        peo, treewidth = gm.get_peo(reduced_graph)

        graph_parallel, label_dict = gm.relabel_graph_nodes(
            reduced_graph, dict(zip(range(1, len(peo) + 1), peo))
        )

        mem_cost, flop_cost = gm.cost_estimator(graph_parallel)

        max_mem = sum(mem_cost)
        min_mem = max(mem_cost)
        flops = sum(flop_cost)

        total_mem_max = max_mem * (2**n_var_parallel)
        total_min_mem = min_mem * (2**n_var_parallel)
        total_flops = flops * (2**n_var_parallel)

        results.append((max_mem, min_mem, flops,
                        total_mem_max, total_min_mem,
                        total_flops, treewidth))

    return tuple(zip(*results))


def get_treewidth_vs_parallel_size(filename, metric_function,
                                   start_at=0, stop_at=None,
                                   step_by=1):
    """
    Calculates treewidth vs the number of parallelized
    variables for a given circuit.

    Parameters
    ----------
    filename : str
           input file
    Returns
    -------
          treewidth - treewidth returned by quickBB
    """
    n_qubits, circuit = ops.read_circuit_file(filename)

    buckets, _ = opt.circ2buckets(circuit)
    graph_raw = opt.buckets2graph(buckets)
    n_var_total = graph_raw.number_of_nodes()
    if stop_at is None or stop_at > n_var_total:
        stop_at = n_var_total

    results = []
    for n_var_parallel in range(start_at, stop_at, step_by):
        idx_parallel, reduced_graph = gm.split_graph_by_metric(
            graph_raw, n_var_parallel, metric_fn=metric_function)
        peo, treewidth = gm.get_peo(reduced_graph)

        results.append(treewidth)

    return results


def plot_cost_vs_n_var_parallel(
        filename,
        fig_filename='memory_vs_parallelized_vars.png',
        start_at=0, stop_at=None, step_by=5):
    """
    Plots memory requirement with respect to the number of
    parallelized variables
    """
    costs = get_cost_vs_parallel_size(filename, start_at=start_at,
                                      stop_at=stop_at, step_by=step_by)
    x_range = list(range(start_at, start_at+len(costs[0])*step_by, step_by))
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 6))

    axes[0].semilogy(x_range, costs[0], label='per node')
    axes[0].semilogy(x_range, costs[3], label='total')
    axes[0].set_xlabel('parallelized variables')
    axes[0].set_ylabel('memory (in doubles)')
    axes[0].set_title('Maximal memory requirement')
    axes[0].legend()

    axes[1].semilogy(x_range, costs[1], label='per node')
    axes[1].semilogy(x_range, costs[4], label='total')
    axes[1].set_xlabel('parallelized variables')
    axes[1].set_ylabel('memory (in doubles)')
    axes[1].set_title('Minimal memory requirement')
    axes[1].legend()

    axes[2].semilogy(x_range, costs[5], 'b-', label='flops')
    axes[1].set_xlabel('parallelized variables')
    axes[2].set_ylabel('flops')
    axes[2].set_title('Flops cost')
    axes[2].legend(loc='upper right')

    ax21 = axes[2].twinx()
    ax21.plot(x_range, costs[6], 'r-', label='treewidth')
    ax21.set_ylabel('treewidth')
    ax21.legend(loc='lower right')

    fig.savefig(fig_filename)


def plot_flops_vs_n_var_parallel(
        filename,
        fig_filename='treewidth_vs_parallelized_vars.png',
        step_by=5):
    """
    Plots treewidth and flop count with respect to the number of
    parallelized variables
    """
    costs = get_cost_vs_parallel_size(filename, step_by)
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(20, 6))
    x_range = list(range(0, len(costs[0])*step_by, step_by))

    ax.plot(x_range, costs[6], 'r-', label='treewidth')
    ax.set_xlabel('Number of parallelized variables')
    ax.set_ylabel('treewidth')
    ax.set_title('Graph treewidth')

    ax2 = ax.twinx()
    ax2.semilogy(x_range, costs[5], 'b-', label='flops')
    ax.set_ylabel('flops')

    ax.legend(loc='lower right')
    ax2.legend(loc='upper right')

    fig.savefig(fig_filename)


def plot_compare_parallelization_strategies(
        filename, fig_filename='compare_strategies.png',
        start_at=0, stop_at=None, step_by=1):
    """
    Compares treewidth reduction strategies
    """
    metric_functions = {
        'degree': gm.get_node_by_degree,
        'betweenness': gm.get_node_by_betweenness,
        'bruteforce (Chen et al.)': gm.get_node_by_mem_reduction
    }

    results = {}
    for name, metric_function in metric_functions.items():
        treewidth_vs_n_var = get_treewidth_vs_parallel_size(
            filename, metric_function=metric_function,
            start_at=start_at, stop_at=stop_at, step_by=step_by)
        results.update({name: treewidth_vs_n_var})

    x_range = list(
        range(
            start_at,
            start_at+len(next(iter(results.values())))*step_by, step_by))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for name in sorted(results.keys()):
        ax.plot(x_range, results[name], label=name)
    ax.set_xlabel('Number of parallelized variables')
    ax.set_ylabel('treewidth')
    ax.set_title('Strategies for node removal')
    ax.legend()

    fig.savefig(fig_filename)


def plot_compare_step(
        filename, fig_filename='compare_step.png',
        steps=[1, 5],
        start_at=0, stop_at=None):
    """
    Compares treewidth reduction with different step
    """
    metric_function = gm.get_node_by_degree

    results = {}
    for step_by in steps:
        treewidth_vs_n_var = get_treewidth_vs_parallel_size(
            filename, metric_function=metric_function,
            start_at=start_at, stop_at=stop_at, step_by=step_by)
        results.update({step_by: treewidth_vs_n_var})

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for step_by, marker in zip(
            sorted(results.keys()),
            itertools.cycle(
                ('.', '*', '+', 'x', '1', '2', '3', '4'))):
        x_range = list(
            range(
                start_at,
                start_at+len(results[step_by])*step_by,
                step_by))
        ax.plot(x_range, results[step_by],
                label='step by {}'.format(step_by), marker=marker)
    ax.set_xlabel('Number of parallelized variables')
    ax.set_ylabel('treewidth')
    ax.set_title('Step size influence')
    ax.legend()

    fig.savefig(fig_filename)


def test_split_with_mem_constraint(filename, constraints, step_by=1):
    """
    Test the mem constraint contraction splitting. Input is read
    from filename, for each constraint the treewidth and attained
    cost is printed
    """
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, _ = opt.circ2buckets(circuit)
    graph_raw = opt.buckets2graph(buckets)
    n_var_total = graph_raw.number_of_nodes()

    results = []
    for mem_constraint in constraints:
        idx_parallel, reduced_graph = gm.split_graph_with_mem_constraint(
             graph_raw, 0, mem_constraint, step_by=step_by)

        peo, treewidth = gm.get_peo(reduced_graph)

        mem_cost, flop_cost = gm.cost_estimator(reduced_graph)
        max_mem = sum(mem_cost)
        flops = sum(flop_cost)

        total_mem_max = max_mem * (2**len(idx_parallel))
        total_flops = flops * (2**len(idx_parallel))

        results.append([max_mem, total_mem_max, flops, total_flops])

    for mem_constraint, res in zip(constraints, results):
        max_mem, total_mem_max, flops, total_flops = res
        log.info('Set memory constraint per node: {:e}'.format(
            mem_constraint))
        log.info('Attained values:')
        log.info(' total:')
        log.info(' memory: {:e} flop: {:e}'.format(
            total_mem_max, total_flops))
        log.info(' per node:')
        log.info(' memory: {:e} flop: {:e}'.format(
            max_mem, flops))

    return results


def collect_costs(
        out_filename,
        grid_sizes=[4, 5],
        depths=list(range(10, 15)),
        path_to_testcases='./test_circuits/inst/cz_v2',
        n_var_parallel=0):
    """
    Calculates costs for test circuits with grid size equal to grid_sizes
    and outputs results to a pandas.DataFrame, and saves to out_filename
    (as pickle). This is a symbolic equivalent (no real evaluation is
    done) of the :py:meth:`collect_timings`.
    """

    try:
        data = pd.read_pickle(out_filename)
    except FileNotFoundError:
        # lays down the structure of data
        data = pd.DataFrame(
            [],
            index=['mem_min', 'mem_max', 'flop', 'treewidth'],
            columns=pd.MultiIndex.from_product(
                [[], []], names=['grid size', 'depth']))

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

            n_qubits, circuit = ops.read_circuit_file(testfile)
            buckets, _ = opt.circ2buckets(circuit)
            graph_raw = opt.buckets2graph(buckets)

            idx_parallel, reduced_graph = gm.split_graph_by_metric(
                graph_raw, n_var_parallel,
                metric_fn=gm.get_node_by_degree)

            peo, treewidth = gm.get_peo(reduced_graph)
            mem_cost, flop_cost = gm.cost_estimator(reduced_graph)

            mem_min = max(mem_cost)
            mem_max = sum(mem_cost)
            flops = sum(flop_cost)

            # Merge current result with the rest
            data[grid_size, depth] = [mem_min, mem_max, flops, treewidth]
            # Save result
            data.to_pickle(out_filename)

    return data


def extract_costs_vs_depth(
        filename, depths,
        grid_size=4, rec_id='max_mem'):
    """
    Extracts timings vs depth from the timings data file
    for a fixed grid_size
    """
    data = pd.read_pickle(filename)

    values = []
    for depth in depths:
        time = data[(grid_size, depth)][rec_id]
        values.append(time)

    return values, depths


def extract_costs_vs_gridsize(
        filename, grid_sizes,
        depth=10, rec_id='flop'):
    """
    Extracts timings vs grid size from the timings data file
    for a fixed depth
    """
    data = pd.read_pickle(filename)

    times = []
    for grid_size in grid_sizes:
        time = data[(grid_size, depth)][rec_id]
        times.append(time)

    return times, grid_sizes


def plot_cost_vs_depth(filename,
                       n_var_parallel=0,
                       fig_filename='cost_vs_depth.png',
                       interactive=False):
    """
    Plots cost estimates vs depth for some number of grid sizes
    Data is loaded from filename
    """
    if not interactive:
        plt.switch_backend('agg')

    grid_sizes = [4, 5]
    depths = range(10, 50)

    # Create empty canvas
    fig, axes = plt.subplots(1, len(grid_sizes), sharey=True,
                             figsize=(24, 6))

    for n, grid_size in enumerate(grid_sizes):
        flop, depths_labels = extract_costs_vs_depth(
            filename, depths, grid_size, rec_id='flop')
        axes[n].semilogy(depths_labels, np.array(flop)*2**n_var_parallel,
                         'b-', label='flops')
        axes[n].set_xlabel(
            'depth')
        axes[n].set_title('{}x{} circuit'.format(grid_size, grid_size))
        axes[n].set_ylabel('flops')
        axes[n].legend(loc='upper left')

        right_ax = axes[n].twinx()
        treewidth, depths_labels = extract_costs_vs_depth(
            filename, depths, grid_size, rec_id='treewidth')
        right_ax.plot(depths_labels, treewidth, 'r-', label='treewidth')
        right_ax.set_ylabel('treewidth')
        right_ax.legend(loc='lower right')

    fig.suptitle('Evaluation cost vs depth of the circuit')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_cost_vs_gridsize(filename,
                          n_var_parallel=0,
                          fig_filename='cost_vs_gridsize.png',
                          interactive=False):
    """
    Plots cost estimates vs gridsize for some number of depths
    Data is loaded from filename
    """
    if not interactive:
        plt.switch_backend('agg')

    grid_sizes = [4, 5]
    depths = range(10, 50)

    # Create empty canvas
    fig, axes = plt.subplots(1, len(grid_sizes), sharey=True,
                             figsize=(12, 6))

    for n, depth in enumerate(depths):
        flop, gridsize_labels = extract_costs_vs_depth(
            filename, grid_sizes, depth, rec_id='flop')
        axes[n].semilogy(gridsize_labels, np.array(flop)*2**n_var_parallel,
                         'b-', label='flops')
        axes[n].set_xlabel(
            'depth')
        axes[n].set_title('{} depth'.format(depth))
        axes[n].set_ylabel('flops')
        axes[n].legend(loc='upper left')

        right_ax = axes[n].twinx()
        treewidth, gridsize_labels = extract_costs_vs_depth(
            filename,  grid_sizes, depth, rec_id='treewidth')
        right_ax.plot(gridsize_labels, treewidth, 'r-', label='treewidth')
        right_ax.set_ylabel('treewidth')
        right_ax.legend(loc='lower right')

    fig.suptitle('Evaluation cost vs depth of the circuit')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_cost_vs_depth_multiple(
        filename,
        n_var_parallel=0,
        fig_filename='cost_vs_depth.png',
        interactive=False):
    """
    Plots cost estimates (per node) vs depth for some
    number of grid sizes and some number of depths for each grid size
    """
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    if not interactive:
        plt.switch_backend('agg')

    grid_sizes = [6, 7, 8, 9, 10, 11]
    depths_list = [range(50, 69), range(41, 54), range(35, 45),
                   range(31, 40), range(27, 36), range(25, 32)
    ]

    # Create empty canvas
    fig, ax = plt.subplots(1, 1, sharey=True,
                           figsize=(12, 12))

    jet = cm = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=len(grid_sizes))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    for n, (grid_size, depths) in enumerate(zip(grid_sizes, depths_list)):
        flop, depths_labels = extract_costs_vs_depth(
            filename, depths, grid_size, rec_id='flop')
        ax.semilogy(depths_labels, flop, color=scalarMap.to_rgba(n), marker='o',
                    label=f'n = {grid_size}')
        ax.set_xlabel(
            'Depth')
        ax.set_title('Predicted runtimes for the Qtree simulator')
        ax.set_ylabel('Predicted runtime (s)')
        ax.legend(loc='upper right')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


if __name__ == "__main__":
    n = 4
    d = 20
    idx = 2
    step_by = 1

    # plot_cost_vs_n_var_parallel(
    #     filename=f'test_circuits/inst/cz_v2/{n}x{n}/inst_{n}x{n}_{d}_{idx}.txt',
    #     fig_filename=f'memory_vs_parallelized_vars_{n}x{n}_{d}.png',
    #     start_at=5, stop_at=300, step_by=step_by
    # )
    # plot_compare_parallelization_strategies(
    #     filename=f'test_circuits/inst/cz_v2/{n}x{n}/inst_{n}x{n}_{d}_{idx}.txt',
    #     fig_filename=f'parallelization_strategies_{n}x{n}_{d}.png',
    #     start_at=0, stop_at=None, step_by=1
    # )

    # plot_compare_step(
    #     filename=f'test_circuits/inst/cz_v2/{n}x{n}/inst_{n}x{n}_{d}_{idx}.txt',
    #     fig_filename=f'compare_step_{n}x{n}_{d}.png',
    #     start_at=0, stop_at=24, steps=[1, 3]
    # )
    # constraints = [1e2, 1e3, 1e4]
    # test_split_with_mem_constraint(
    #     filename=f'test_circuits/inst/cz_v2/{n}x{n}/inst_{n}x{n}_{d}_{idx}.txt',
    #     constraints=constraints, step_by=5)

    n_var_parallel = 17
    collect_costs(f'output/cost_estimate_{n_var_parallel}.p',
                  grid_sizes=[5, 6, 7, 8, 9, 10, 11, 12],
                  depths=list(range(10, 70)),
                  n_var_parallel=n_var_parallel,
    )
    plot_cost_vs_depth(f'output/cost_estimate_{n_var_parallel}.p',
                       fig_filename=f'cost_vs_depth_{n_var_parallel}.png'
    )
