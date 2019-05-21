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

import matplotlib.pyplot as plt

from src.logger_setup import log
from functools import partial

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

# controls default text sizes
plt.rc('font', size=SMALL_SIZE)
# fontsize of the axes title
plt.rc('axes', titlesize=BIGGER_SIZE)
# fontsize of the x and y labels
plt.rc('axes', labelsize=MEDIUM_SIZE)
# fontsize of the tick labels
plt.rc('xtick', labelsize=SMALL_SIZE)
# fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)
# legend fontsize
plt.rc('legend', fontsize=MEDIUM_SIZE)
# fontsize of the figure title
plt.rc('figure', titlesize=BIGGER_SIZE)


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
    Wow, new method works!
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
        # Now calculate the real treewidth, as it may be higher then the value
        # returned by quickbb
        treewidth_original = gm.get_treewidth_from_peo(graph, peo_original)
        peo_upperbound, treewidth_upperbound = gm.get_upper_bound_peo(graph)

        # Magic procedure to transform peo using chordal graphs
        peo_new = gm.get_equivalent_peo(
            graph, peo_original, free_variables)
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


def get_cost_vs_amp_subset_size_parallel(
        filename, step_by=1, start_at=0, stop_at=None,
        n_var_parallel=0, peo_transform_fun=gm.get_equivalent_peo,
        splitting_fun=gm.split_graph_by_metric, out_filename=''):
    """
    Calculates memory cost vs the number of calculated amplitudes
    for a given circuit. Amplitudes are calculated in subsets up to the
    full state vector

    Parameters
    ----------
    filename : str
           input file
    start_at : int, optional
           number of full qubits to start. Default 0
    stop_at : int, optional
           number of full qubits to stop at. Default all qubits
    step_by : int, optional
           add this number of full qubits in the next result. Default 1
    n_var_parallel : int, optional
           number of variables to parallelize over. Default 0
    peo_transform_fun : function, default gm.get_equivalent_peo
           function used to transform PEO. It's signature should be
           f(graph, peo, end_indices)  -> new_peo
    out_filename : str
           Pandas pickle for possible output of the results
    Returns
    -------
          max_mem - maximal memory (if all intermediates are kept)
                    per task
          min_mem - minimal possible memory for the algorithm per task
          flops - flops count per task
          total_mem_max - total amount of memory for all tasks
          total_min_mem - total minimal amount of memory for all tasks
          total_flops - total number of flops
          max_mem_best - maximal memory if PEO would be optimal
                         per task
          min_mem_best - minimal memory if PEO would be optimal
                         per task
          flops_best - flops count if PEO would be optimal
          treewidth - treewidth returned by quickBB
          treewidth_best - treewidth if PEO would be optimal
          av_flop_per_mem - average memory access per flop
    """
    # Load graph and get the number of nodes
    n_qubits, buckets, free_vars = opt.read_buckets(filename)

    if stop_at is None or stop_at > n_qubits:
        stop_at = n_qubits + 1

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

        # Remove n_var_parallel variables from graph
        # Currently, we do not remove output qubit variables
        # This corresponds to the calculation of amplitude tensor
        # in full for each subtask (not splitting it between
        # different subtasks)
        if n_var_parallel > graph.number_of_nodes() - len(free_variables):
            n_var_parallel = graph.number_of_nodes() - len(free_variables)

        idx_parallel, reduced_graph = splitting_fun(
            graph, n_var_parallel, forbidden_nodes=free_variables)

        # This is the best possible treewidth.
        # What will our method produce?
        peo_best, treewidth_best = gm.get_peo(reduced_graph)
        # Recalculate treewidth as quickbb returns lower values
        # than we actually have
        treewidth_best = gm.get_treewidth_from_peo(reduced_graph, peo_best)

        peo = gm.get_equivalent_peo(
            reduced_graph, peo_best, free_variables)
        treewidth = gm.get_treewidth_from_peo(reduced_graph, peo)

        graph_final, label_dict = gm.relabel_graph_nodes(
            reduced_graph, dict(zip(peo, range(1, len(peo) + 1)))
        )

        mem_cost, flop_cost = gm.cost_estimator(graph_final)

        max_mem = sum(mem_cost)
        min_mem = max(mem_cost)
        flops = sum(flop_cost)

        total_mem_max = max_mem * (2**n_var_parallel)
        total_min_mem = min_mem * (2**n_var_parallel)
        total_flops = flops * (2**n_var_parallel)

        graph_best, label_dict = gm.relabel_graph_nodes(
            reduced_graph, dict(zip(peo_best, range(1, len(peo_best) + 1)))
        )

        mem_cost_best, flop_cost_best = gm.cost_estimator(graph_best)

        max_mem_best = sum(mem_cost_best)
        min_mem_best = max(mem_cost_best)
        flops_best = sum(flop_cost_best)

        total_mem_max_best = max_mem_best * (2**n_var_parallel)
        total_min_mem_best = min_mem_best * (2**n_var_parallel)
        total_flops_best = flops_best * (2**n_var_parallel)

        flop_per_mem = [flop / mem for mem, flop
                        in zip(mem_cost, flop_cost)]
        av_flop_per_mem = sum(flop_per_mem) / len(flop_per_mem)

        results.append((max_mem, min_mem, flops,
                        total_mem_max, total_min_mem,
                        total_flops,
                        max_mem_best, min_mem_best,
                        flops_best,
                        total_mem_max_best,
                        total_min_mem_best,
                        total_flops_best,
                        treewidth,
                        treewidth_best,
                        av_flop_per_mem))

    if len(out_filename) > 0:
        data = pd.DataFrame(
            results,
            columns=['max_mem', 'min_mem', 'flops',
                     'total_mem_max', 'total_min_mem',
                     'total_flops', 'max_mem_best', 'min_mem_best',
                     'flops_best', 'total_mem_max_best',
                     'total_min_mem_best', 'total_flops_best',
                     'treewidth', 'treewidth_best',
                     'av_flop_per_mem'],
            )
        data['n_var_parallel'] = list(range(start_at, stop_at, step_by))
        pd.to_pickle(data, out_filename)

    return tuple(zip(*results))


def plot_cost_vs_amp_subset_size(
        filename,
        fig_filename='flops_vs_amp_subset_size.png',
        start_at=0, stop_at=None, step_by=5,
        n_var_parallel=0,
        peo_transform_fun=gm.get_equivalent_peo):
    """
    Plots cost estimate for the evaluation of subsets of
    amplitudes. This is an old version which compared strategies
    """
    costs = get_cost_vs_amp_subset_size_parallel(
        filename, start_at=start_at,
        stop_at=stop_at, step_by=step_by,
        n_var_parallel=n_var_parallel,
        peo_transform_fun=peo_transform_fun)
    (max_mem, min_mem, flops,
     total_mem_max, total_min_mem, total_flops,
     max_mem_best, min_mem_best, flops_best,
     total_mem_max_best, total_min_mem_best,
     total_flops_best, treewidth, treewidth_best,
     av_flop_per_mem) = costs

    x_range = list(range(start_at,
                         start_at+len(max_mem)*step_by, step_by))
    fig, axes = plt.subplots(1, 3, sharey=False, figsize=(18, 6))

    axes[0].semilogy(x_range, min_mem, 'mo', label='transformed order')
    axes[0].semilogy(x_range, min_mem_best, 'b-', label='best possible')
    num_amplitudes = [2**x for x in x_range]
    mem_one_amplitude_equivalent = [
        min_mem[0] * num_amps for num_amps in num_amplitudes]
    axes[0].semilogy(x_range,
                     mem_one_amplitude_equivalent,
                     'r-', label='1 amp at a time')
    axes[0].set_xlabel('number of full qubits')
    axes[0].set_ylabel('memory (in doubles)')
    axes[0].set_title('Minimal memory requirement')
    axes[0].legend()

    axes[1].semilogy(x_range, flops, 'mo', label='transformed order')
    axes[1].semilogy(x_range, flops_best, 'b-', label='best possible')
    num_amplitudes = [2**x for x in x_range]
    flops_one_amplitude_equivalent = [
        flops[0] * num_amps for num_amps in num_amplitudes]
    axes[1].semilogy(
        x_range,
        flops_one_amplitude_equivalent, 'r-', label='1 amp at a time')

    axes[1].set_xlabel('number of full qubits')
    axes[1].set_ylabel('flops')
    axes[1].set_title('Flops cost')
    axes[1].legend(loc='lower right')

    axes[2].plot(x_range, treewidth, 'mo',
                 label='transformed order')
    axes[2].plot(x_range, treewidth_best, 'b-', label='treewidth best')
    axes[2].set_xlabel('number of full qubits')
    axes[2].set_ylabel('treewidth')
    axes[2].legend(loc='upper left')

    fig.savefig(fig_filename)


def plot_cost_vs_amp_subset_size2(
        filename,
        fig_filename='flops_mem_vs_amp_subset_size.png',
        start_at=0, stop_at=None, step_by=5,
        n_var_parallel=0,
        peo_transform_fun=gm.get_equivalent_peo):
    """
    Plots cost estimates for the evaluation of subsets of
    amplitudes. This is a new improved version
    """
    costs = get_cost_vs_amp_subset_size_parallel(
        filename, start_at=start_at,
        stop_at=stop_at, step_by=step_by,
        n_var_parallel=n_var_parallel,
        peo_transform_fun=peo_transform_fun,
        out_filename=fig_filename+'_out.p')
    (max_mem, min_mem, flops,
     total_mem_max, total_min_mem, total_flops,
     max_mem_best, min_mem_best, flops_best,
     total_mem_max_best, total_min_mem_best,
     total_flops_best, treewidth, treewidth_best,
     av_flop_per_mem) = costs

    x_range = list(range(start_at,
                         start_at+len(max_mem)*step_by, step_by))
    fig, axes = plt.subplots(1, 3, sharey=False, figsize=(18, 6))

    axes[0].semilogy(x_range, min_mem, 'm-', label='batches of amplitudes')
    num_amplitudes = [2**x for x in x_range]
    mem_one_amplitude_equivalent = [
        min_mem[0] for num_amps in num_amplitudes]
    axes[0].semilogy(x_range,
                     mem_one_amplitude_equivalent,
                     'r-', label='1 amplitude at a time')
    axes[0].set_xlabel('number of full qubits')
    axes[0].set_ylabel('memory per task (in doubles)')
    axes[0].set_title('Minimal memory requirement')
    axes[0].legend()

    axes[1].semilogy(x_range, flops, 'm-', label='batches of amplitudes')
    num_amplitudes = [2**x for x in x_range]
    flops_one_amplitude_equivalent = [
        flops[0] * num_amps for num_amps in num_amplitudes]
    axes[1].semilogy(
        x_range,
        flops_one_amplitude_equivalent, 'r-', label='1 amplitude at a time')

    axes[1].set_xlabel('number of full qubits')
    axes[1].set_ylabel('total flops (all tasks)')
    axes[1].set_title('Flops cost')
    axes[1].legend(loc='upper left')

    # tw_one_amplitude_equivalent = [
    #     treewidth[0] for num_amps in num_amplitudes]
    # axes[2].plot(x_range, tw_one_amplitude_equivalent, 'r-',
    #              label='1 amplitude at a time')
    # axes[2].plot(x_range, treewidth, 'm-',
    #              label='batches of amplitudes')
    # axes[2].set_xlabel('number of full qubits')
    # axes[2].set_ylabel('treewidth')
    # axes[2].legend(loc='upper left')

    axes[2].plot(x_range, av_flop_per_mem, 'm-',
                 label='batches of amplitudes')
    axes[2].set_xlabel('number of full qubits')
    axes[2].set_ylabel('flops per memory access')
    axes[2].legend(loc='upper left')

    fig.savefig(fig_filename)


def plot_flops_vs_amp_subset_size_multiple(
        filename_base,
        grid_sizes=[5, 6],
        fig_filename='flops_vs_amp_subset_size.eps'):
    """
    Plots cost estimates for the evaluation of subsets of
    amplitudes. This is a new improved version
    """
    fig, axes = plt.subplots(1, len(grid_sizes),
                             sharey=False, figsize=(6*len(grid_sizes), 6))
    for ii, grid_size in enumerate(grid_sizes):
        data = pd.read_pickle(filename_base
                              + f'_{grid_size}x{grid_size}.p')
        n_var_parallel = data['n_var_parallel']
        flops = data['flops']
        axes[ii].semilogy(n_var_parallel, flops, 'm-',
                          label='batches of amplitudes')

        num_amplitudes = [2**x for x in n_var_parallel]

        flops_one_amplitude_equivalent = [
            flops[0] * num_amps for num_amps in num_amplitudes]
        axes[ii].semilogy(
            n_var_parallel,
            flops_one_amplitude_equivalent, 'r-',
            label='1 amplitude at a time')

        axes[ii].set_xlabel('number of full qubits')
        axes[ii].set_title(f'{grid_size}x{grid_size}')
        axes[ii].set_ylabel('total flops (all tasks)')
        axes[ii].legend(loc='upper left')

    # fig.suptitle('Flops cost')
    fig.savefig(fig_filename)


def plot_mem_vs_amp_subset_size_multiple(
        filename_base,
        grid_sizes=[5, 6],
        fig_filename='mem_vs_amp_subset_size.eps'):
    """
    Plots cost estimates for the evaluation of subsets of
    amplitudes. This is a new improved version
    """
    fig, axes = plt.subplots(1, len(grid_sizes),
                             sharey=False, figsize=(6*len(grid_sizes), 6))
    for ii, grid_size in enumerate(grid_sizes):
        data = pd.read_pickle(filename_base
                              + f'_{grid_size}x{grid_size}.p')
        n_var_parallel = data['n_var_parallel']
        min_mem = data['min_mem']
        axes[ii].semilogy(n_var_parallel, min_mem, 'm-',
                          label='batches of amplitudes')

        num_amplitudes = [2**x for x in n_var_parallel]

        mem_one_amplitude_equivalent = [
            min_mem[0] for num_amps in num_amplitudes]
        axes[ii].semilogy(
            n_var_parallel,
            mem_one_amplitude_equivalent, 'r-',
            label='1 amplitude at a time')

        axes[ii].set_xlabel('number of full qubits')
        axes[ii].set_title(f'{grid_size}x{grid_size}')
        axes[ii].set_ylabel('memory per task (in floats)')
        axes[ii].legend(loc='upper left')

    # fig.suptitle('Minimal memory requirement')
    fig.savefig(fig_filename)


def plot_fpm_vs_amp_subset_multiple(
        filename_base,
        grid_sizes=[5, 6],
        fig_filename='fpm_vs_amp_subset_size.eps'):
    """
    Plots cost estimates for the evaluation of subsets of
    amplitudes. This is a new improved version
    """
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    fig, ax = plt.subplots(1, 1,
                           sharey=False, figsize=(6, 6))

    jet = plt.get_cmap('rainbow')
    cNorm  = colors.Normalize(vmin=0, vmax=len(grid_sizes))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    for ii, grid_size in enumerate(grid_sizes):
        data = pd.read_pickle(filename_base
                              + f'_{grid_size}x{grid_size}.p')
        n_var_parallel = data['n_var_parallel']
        av_flop_per_mem = data['av_flop_per_mem']
        ax.plot(n_var_parallel, av_flop_per_mem,
                color=scalarMap.to_rgba(ii), marker=None,
                label=f'{grid_size}x{grid_size}')


    ax.set_xlabel(f'number of full qubits')
    ax.set_ylabel('flops per memory access')
    ax.legend(loc='lower right')

    fig.savefig(fig_filename)


def dump_nonequivalent_treewidth_graphs(
        filename, out_filename,
        step_by=1, start_at=0, stop_at=None,
        n_var_parallel=0):
    """
    Performs an experiment on peo transformation.
    Dumps the graphs and peo where treewidth of the
    transformed graph does not match the original

    It looks like graph_model.get_treewidth_from_peo()
    returns the treewidth which is larger than quickbb reports
    This larger treewidth does not change after the
    transformation.
    """
    try:
        data = pd.read_pickle(out_filename)
    except FileNotFoundError:
        # lays down the structure of data
        data = pd.DataFrame(
            [],
            index=['clique_vertices', 'graph', 'graph_chordal', 'peo',
                     'new_peo', 'treewidth', 'new_treewidth'],
            columns=['clique_size']
        )

    # Load graph and get the number of nodes
    n_qubits, buckets, free_vars = opt.read_buckets(filename)

    if stop_at is None or stop_at > n_qubits:
        stop_at = n_qubits + 1

    for n_free_qubits in range(start_at, stop_at, step_by):
        free_qubits = range(n_free_qubits)

        # Rebuild the graph with a given number of free qubits
        n_qubits, buckets, free_variables = opt.read_buckets(
            filename,
            free_qubits=free_qubits)
        graph_raw = opt.buckets2graph(buckets)

        # Make a clique on the nodes we do not want to remove
        graph = gm.make_clique_on(graph_raw, free_variables)

        # Remove n_var_parallel variables from graph
        if n_var_parallel > graph.number_of_nodes() - len(free_variables):
            n_var_parallel = graph.number_of_nodes() - len(free_variables)

        idx_parallel, reduced_graph = gm.split_graph_by_metric(
            graph, n_var_parallel, forbidden_nodes=free_variables)

        # This is the best possible treewidth.
        peo_best, treewidth_best = gm.get_peo(reduced_graph)

        peo = gm.get_equivalent_peo(
            reduced_graph, peo_best, free_variables)
        treewidth = gm.get_treewidth_from_peo(reduced_graph, peo)

        # Chordal graph
        graph_chordal = gm.get_fillin_graph(reduced_graph, peo_best)

        # Merge current result with the rest
        data[n_free_qubits] = [
            free_variables, reduced_graph, graph_chordal, peo_best,
            peo, treewidth_best, treewidth]

        data.to_pickle(out_filename)

    return data


def test_mcs_with_graphs():
    """
    This tests the MCS algorithm with hand-made graphs
    """
    g1 = nx.Graph()
    g1.add_edges_from([(1, 2), (2, 3), (3, 1),
                       (3, 4), (4, 5), (5, 6),
                       (6, 4), (6, 7), (7, 8),
                       (8, 9), (9, 7)])
    g1 = gm.wrap_general_graph_for_qtree(g1)
    gm.draw_graph(g1, 'testgr.png')

    peo = gm.maximum_cardinality_search(
        g1, last_clique_vertices=[4, 6, 5])
    tw = gm.get_treewidth_from_peo(g1, peo)


if __name__ == "__main__":
    # test_minfill_heuristic()
    # test_reordering_hypothesis(['test_circuits/inst/cz_v2/4x4/inst_4x4_10_0.txt'])

    # plot_cost_vs_amp_subset_size(
    #     'test_circuits/inst/cz_v2/6x6/inst_6x6_25_0.txt',
    #     fig_filename='costs_amp_subset_6x6_25.png',
    #     start_at=0, step_by=1
    # )

    # plot_cost_vs_amp_subset_size(
    #     'test_circuits/inst/cz_v2/7x7/inst_7x7_39_0.txt',
    #     fig_filename='taihulight_amp_subset_7x7_39.png',
    #     start_at=0, step_by=1, n_var_parallel=23
    # )

    # plot_cost_vs_amp_subset_size2(
    #     'test_circuits/inst/cz_v2/9x9/inst_9x9_25_0.txt',
    #     fig_filename='costs_amp_subset2_9x9_25.eps',
    #     start_at=0, step_by=1
    # )

    plot_mem_vs_amp_subset_size_multiple(
        'costs_amp_subset2',
        grid_sizes=[7, 8, 9],
        fig_filename='mem_vs_amp_subset_size.eps')

    plot_flops_vs_amp_subset_size_multiple(
        'costs_amp_subset2',
        grid_sizes=[7, 8, 9],
        fig_filename='flops_vs_amp_subset_size.eps')

    plot_fpm_vs_amp_subset_multiple(
        'costs_amp_subset2',
        grid_sizes=[7, 8, 9],
        fig_filename='fpm_vs_amp_subset_size.eps')

    # dump_nonequivalent_treewidth_graphs(
    #     'test_circuits/inst/cz_v2/6x6/inst_6x6_25_0.txt',
    #     out_filename='tw_graphs.p',
    #     start_at=5, step_by=1, stop_at=16
    # )

    # Taihuligh single amplitude estimate
    # costs = get_cost_vs_amp_subset_size_parallel(
    #     'test_circuits/inst/cz_v2/8x8/inst_8x8_44_0.txt',
    #     start_at=0, stop_at=1, step_by=1, n_var_parallel=23,
    #     splitting_fun=partial(gm.split_graph_dynamic_greedy,
    #                           greedy_step_by=1)
    # )
    # (max_mem, min_mem, flops,
    #  total_mem_max, total_min_mem, total_flops,
    #  max_mem_best, min_mem_best, flops_best,
    #  total_mem_max_best, total_min_mem_best,
    #  total_flops_best, treewidth, treewidth_best,
    #  av_flop_per_mem) = costs

    # print(min_mem[0]/2**30)
    # print(np.log2(flops[0]))
    # print(flops[0] / 10**10)
