"""
In this file we test calculation of subsets of amplitudes
versus calculating amplitudes one by one
"""

import numpy as np
import networkx as nx
import pandas as pd
import random
import copy

import src.optimizer as opt
import src.graph_model as gm

from src.logger_setup import log
from matplotlib import pyplot as plt


def test_minfill_heuristic():
    """
    Tests minfill heuristic using quickbb algorithm
    """
    # Test 1: path graph with treewidth 1
    print('Test 1. Path graph')
    graph = gm.wrap_general_graph_for_qtree(
        nx.path_graph(8)
    )

    peo_minfill, tw_minfill = gm.get_upper_bound_peo(graph)
    peo_quickbb, tw_quickbb = gm.get_peo(graph)
    print(f'minfill treewidth: {tw_minfill} quickbb treewidth: {tw_quickbb}')
    print(f'minfill peo: {peo_minfill}')
    print(f'quickbb peo: {peo_quickbb}')

    # Test 2: complete graph with treewidth n-1
    print('Test 2. Complete graph')
    graph = gm.wrap_general_graph_for_qtree(
        nx.complete_graph(8)
    )

    peo_minfill, tw_minfill = gm.get_upper_bound_peo(graph)
    peo_quickbb, tw_quickbb = gm.get_peo(graph)
    print(f'minfill treewidth: {tw_minfill} quickbb treewidth: {tw_quickbb}')
    print(f'minfill peo: {peo_minfill}')
    print(f'quickbb peo: {peo_quickbb}')

    # Test 3: complicated graphs with indefinite treewidth
    print('Test 3. Probabilistic graph')
    graph = gm.wrap_general_graph_for_qtree(
        gm.generate_random_graph(50, 300)
    )

    peo_minfill, tw_minfill = gm.get_upper_bound_peo(graph)
    peo_quickbb, tw_quickbb = gm.get_peo(graph)
    print(f'minfill treewidth: {tw_minfill} quickbb treewidth: {tw_quickbb}')
    print(f'minfill peo: {peo_minfill}')
    print(f'quickbb peo: {peo_quickbb}')


def eliminate_nodes_from(graph, peo_partial):
    """
    Eliminates nodes given by the list peo partial
    """
    for node in peo_partial:
        gm.eliminate_node(graph, node, self_loops=False)

def test_reordering_hypothesis(filenames):
    """
    Test if the reordering hypotesis holds.
    Hint: it fails in the current version
    """

    for filename in filenames:
        n_qubits, graph = gm.read_graph(filename)
        # n_free_variables = random.randint(1, n_qubits-1)
        n_free_variables = 4
        # free_qubits = np.random.choice(range(n_qubits),
        #                                n_free_variables, replace=False)
        free_qubits = [8, 15, 4, 0]
        n_qubits, buckets, free_variables = opt.read_buckets(
            filename,
            free_qubits=free_qubits)

        graph_initial = opt.buckets2graph(buckets)
        graph = gm.make_clique_on(graph_initial, free_variables)

        peo_original, treewidth_original = gm.get_peo(graph)
        peo_upperbound, treewidth_upperbound = gm.get_upper_bound_peo(graph)

        peo_new = gm.get_equivalent_peo(peo_original, free_variables)
        treewidth_new = gm.get_treewidth_from_peo(graph, peo_new)

        # Check if we do not screw up anything with quickbb
        treewidth_check = gm.get_treewidth_from_peo(graph, peo_original)
        # What if we reverse PEO?
        treewidth_reverse = gm.get_treewidth_from_peo(
            graph, reversed(peo_original))

        # What if we eliminate nodes in the end clique,
        # and then calculate PEO? Lemma 16 in Boadlander
        graph_elim = copy.deepcopy(graph)
        for node in free_variables:
            gm.eliminate_node(graph_elim, node, self_loops=False)
        peo_elim, treewidth_elim = gm.get_peo(graph_elim)
        peo_elim_all = peo_elim + free_variables
        treewidth_elim_all = gm.get_treewidth_from_peo(
            graph, peo_elim_all)

        # What if we delete nodes in the end clique,
        # and then calculate PEO?
        graph_del = copy.deepcopy(graph)
        graph_del.remove_nodes_from(free_variables)

        peo_del, treewidth_del = gm.get_peo(graph_del)
        peo_del_all = peo_del + free_variables
        treewidth_del_all = gm.get_treewidth_from_peo(
            graph, peo_del_all)

        print(f' file: {filename} n_free_vars: {n_free_variables}')
        print(f' free_qubits: {free_qubits}')
        print(f' free_variables: {free_variables}')
        print(f' peo_orig: {peo_original}')
        print(f' peo_new: {peo_new}')
        print(f' tw_orig: {treewidth_original} tw_new : {treewidth_new} tw_check: {treewidth_check}')
        print(f' tw_upper: {treewidth_upperbound}')
        print(f' tw_reverse: {treewidth_reverse}')
        print(f' tw_elim_all: {treewidth_elim_all}')
        print(f' tw_del_all: {treewidth_del_all}')

        if treewidth_new == treewidth_original:
            print('OK')
        else:
            print('FAIL')


def get_cost_vs_amp_subset_size(filename, step_by=1, start_at=0, stop_at=None):
    """
    Calculates memory cost vs the number of calculated amplitudes
    for a given circuit. Amplitudes are calculated in subsets up to the
    full state vector

    Parameters
    ----------
    filename : str
           input file
    Returns
    -------
          max_mem - maximal memory (if all intermediates are kept)
          min_mem - minimal possible memory for the algorithm
          max_mem_best - maximal memory if PEO would be optimal
          min_mem_best - minimal memory if PEO would be optimal
          treewidth - treewidth returned by quickBB
          av_flop_per_mem - average memory access per flop
    """
    # Load graph and get the number of nodes
    n_qubits, buckets, free_vars = opt.read_buckets(filename)

    if stop_at is None or stop_at > n_qubits:
        stop_at = n_qubits

    results = []
    for n_free_qubits in range(start_at, stop_at, step_by):
        free_qubits = range(n_free_qubits)

        # Rebuild the graph with a given number of free qubits
        n_qubits, buckets, free_variables = opt.read_buckets(
            filename,
            free_qubits=free_qubits)
        graph_raw = opt.buckets2graph(buckets)

        # Make a clique on the nodes we do not want to remove
        graph = gm.make_clique_on(graph_raw, free_variables)

        # This is the best possible treewidth.
        # Our method of PEO transformation yields larger values
        peo_best, treewidth_best = gm.get_peo(graph)

        peo = gm.get_equivalent_peo(peo_best, free_variables)
        treewidth = gm.get_treewidth_from_peo(graph, peo)

        graph_final, label_dict = gm.relabel_graph_nodes(
            graph, dict(zip(peo, range(1, len(peo) + 1)))
        )

        mem_cost, flop_cost = gm.cost_estimator(graph_final)

        max_mem = sum(mem_cost)
        min_mem = max(mem_cost)
        flops = sum(flop_cost)

        graph_best, label_dict = gm.relabel_graph_nodes(
            graph, dict(zip(peo_best, range(1, len(peo_best) + 1)))
        )

        mem_cost_best, flop_cost_best = gm.cost_estimator(graph_best)

        max_mem_best = sum(mem_cost_best)
        min_mem_best = max(mem_cost_best)
        flops_best = sum(flop_cost_best)

        flop_per_mem = [flop / mem for mem, flop
                        in zip(mem_cost, flop_cost)]
        av_flop_per_mem = sum(flop_per_mem) / len(flop_per_mem)

        results.append((max_mem, min_mem, flops,
                        max_mem_best, min_mem_best,
                        flops_best, treewidth,
                        treewidth_best,
                        av_flop_per_mem))

    return tuple(zip(*results))


def plot_cost_vs_amp_subset_size(
        filename,
        fig_filename='flops_vs_amp_subset_size.png',
        start_at=0, stop_at=None, step_by=5):
    """
    Plots cost estimate for the evaluation of subsets of
    amplitudes
    """
    costs = get_cost_vs_amp_subset_size(filename, start_at=start_at,
                                        stop_at=stop_at, step_by=step_by)
    x_range = list(range(start_at, start_at+len(costs[0])*step_by, step_by))
    fig, axes = plt.subplots(1, 3, sharey=False, figsize=(18, 6))

    # axes[0].semilogy(x_range, costs[0], label='per node')
    # axes[0].semilogy(x_range, costs[3], label='total')
    # axes[0].set_xlabel('parallelized variables')
    # axes[0].set_ylabel('memory (in doubles)')
    # axes[0].set_title('Maximal memory requirement')
    # axes[0].legend()

    axes[0].semilogy(x_range, costs[1], 'm-', label='current')
    axes[0].semilogy(x_range, costs[4], 'b-', label='best')
    num_amplitudes = [2**x for x in x_range]
    mem_one_amplitude_equivalent = [costs[4][0] * num_amps for num_amps in num_amplitudes] 
    axes[0].semilogy(x_range, mem_one_amplitude_equivalent, 'r-', label='1 amp at a time')
    axes[0].set_xlabel('number of full qubits')
    axes[0].set_ylabel('memory (in doubles)')
    axes[0].set_title('Minimal memory requirement')
    axes[0].legend()

    axes[1].semilogy(x_range, costs[2], 'm-', label='current')
    axes[1].semilogy(x_range, costs[5], 'b-', label='best')
    num_amplitudes = [2**x for x in x_range]
    flops_one_amplitude_equivalent = [costs[5][0] * num_amps for num_amps in num_amplitudes]
    axes[1].semilogy(x_range, flops_one_amplitude_equivalent, 'r-', label='1 amp at a time')

    axes[1].set_xlabel('number of full qubits')
    axes[1].set_ylabel('flops')
    axes[1].set_title('Flops cost')
    axes[1].legend(loc='lower right')

    axes[2] = axes[2].twinx()
    axes[2].plot(x_range, costs[6], 'm-', label='treewidth current')
    axes[2].plot(x_range, costs[7], 'b-', label='treewidth best')
    axes[2].set_xlabel('number of full qubits')
    axes[2].set_ylabel('treewidth')
    axes[2].legend(loc='upper left')

    fig.savefig(fig_filename)


if __name__ == "__main__":
    #test_minfill_heuristic()
    #test_reordering_hypothesis(['test_circuits/inst/cz_v2/4x4/inst_4x4_10_0.txt'])
    # plot_cost_vs_amp_subset_size(
    #     'test_circuits/inst/cz_v2/6x6/inst_6x6_25_0.txt',
    #     fig_filename='costs_amp_subset_6x6_25.png',
    #     start_at=0, step_by=1
    # )
    plot_cost_vs_amp_subset_size(
        'test_circuits/inst/cz_v2/5x5/inst_5x5_25_0.txt',
        fig_filename='costs_amp_subset_5x5_25.png',
        start_at=0, step_by=1
    )
