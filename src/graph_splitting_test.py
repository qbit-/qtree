"""
This module is for the experimentation with splitting
graphs in different ways to reduce the computational effort
of the tensor contraction
"""
import numpy as np
import pandas as pd

import src.operators as ops
import src.optimizer as opt
import src.graph_model as gm

from src.logger_setup import log
from matplotlib import pyplot as plt


def get_cost_vs_parallel_size(filename, step_by=1, start_at=0, stop_at=None):
    """
    Calculates memory cost (per node) vs the number of parallelized
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
        (peo, treewidth,
         idx_parallel, reduced_graph) = gm.get_peo_parallel_by_metric(
             graph_raw, n_var_parallel)

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
        (peo, treewidth,
         idx_parallel, reduced_graph) = gm.get_peo_parallel_by_metric(
             graph_raw, n_var_parallel, metric_fn=metric_function)

        results.append(treewidth)

    return results


def plot_cost_vs_n_var_parallel(
        filename,
        fig_filename='memory_vs_parallelized_vars.png', step_by=5):
    """
    Plots memory requirement with respect to the number of
    parallelized variables
    """
    costs = get_cost_vs_parallel_size(filename, step_by)
    x_range = list(range(0, len(costs[0])*step_by, step_by))
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
    Plots treewdth and flop count with respect to the number of
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
        'betweenness': gm.get_node_by_betweenness
    }

    results = {}
    for name, metric_function in metric_functions.items():
        treewdth_vs_n_var = get_treewidth_vs_parallel_size(
            filename, metric_function=metric_function,
            start_at=start_at, stop_at=stop_at, step_by=step_by)
        results.update({name: treewdth_vs_n_var})

    x_range = list(
        range(
            start_at, len(results.values().__iter__())*step_by, step_by))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for name in sorted(results.keys()):
        ax.plot(x_range, results[name], label=name)
    ax.set_xlabel('Number of parallelized variables')
    ax.set_ylabel('treewidth')
    ax.set_title('Strategies for node removal')
    ax.legend()

    fig.savefig(fig_filename)


if __name__ == "__main__":
    n = 6
    d = 20
    idx = 2
    step_by = 5

    # plot_cost_vs_n_var_parallel(
    #     filename=f'test_circuits/inst/cz_v2/{n}x{n}/inst_{n}x{n}_{d}_{idx}.txt',
    #     fig_filename=f'memory_vs_parallelized_vars_{n}x{n}_{d}.png',
    #     step_by=step_by
    # )
    plot_compare_parallelization_strategies(
        filename=f'test_circuits/inst/cz_v2/{n}x{n}/inst_{n}x{n}_{d}_{idx}.txt',
        fig_filename=f'parallelization_strategies_{n}x{n}_{d}.png',
        start_at=0, stop_at=55, step_by=1
    )
