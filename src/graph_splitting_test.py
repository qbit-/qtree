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


def get_cost_vs_parallel_size(filename):
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
    """
    n_qubits, circuit = ops.read_circuit_file(filename)

    buckets, _ = opt.circ2buckets(circuit)
    graph_raw = opt.buckets2graph(buckets)

    results = []
    for n_var_parallel in range(n_qubits):
        (peo, mem,
         idx_parallel, reduced_graph) = gm.get_peo_parallel_by_metric(
             graph_raw, n_var_parallel)

        graph_parallel, label_dict = gm.relabel_graph_nodes(
            reduced_graph, dict(zip(range(1, len(peo) + 1), peo))
        )

        mem_cost, _ = gm.cost_estimator(graph_parallel)

        max_mem = sum(mem_cost)
        min_mem = max(mem_cost)
        total_mem_max = max_mem * (2**n_var_parallel)
        total_min_mem = min_mem * (2**n_var_parallel)

        results.append((max_mem, min_mem,
                        total_mem_max, total_min_mem))

    return tuple(zip(*results))


def plot_cost_vs_n_var_parallel(
        filename,
        fig_filename='memory_vs_parallelized_vars.png'):
    """
    Plots memory requirement with respect to the number of
    parallelized variables
    """
    costs = get_cost_vs_parallel_size(filename)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
    axes[0].semilogy(costs[0], label='per node')
    axes[0].semilogy(costs[2], label='total')
    axes[0].set_xlabel('parallelized variables')
    axes[0].set_ylabel('memory (in doubles)')
    axes[0].set_title('Maximal memory requirement')
    axes[0].legend()

    axes[1].semilogy(costs[1], label='per node')
    axes[1].semilogy(costs[3], label='total')
    axes[1].set_xlabel('parallelized variables')
    axes[1].set_ylabel('memory (in doubles)')
    axes[1].set_title('Minimal memory requirement')
    axes[1].legend()

    fig.savefig(fig_filename)


if __name__ == "__main__":
    plot_cost_vs_n_var_parallel(
        'test_circuits/inst/cz_v2/4x4/inst_4x4_11_2.txt',
        fig_filename='memory_vs_parallelized_vars_4x4_11.png'
    )

    plot_cost_vs_n_var_parallel(
        'test_circuits/inst/cz_v2/5x5/inst_5x5_20_2.txt',
        fig_filename='memory_vs_parallelized_vars_5x5_20.png'
    )
