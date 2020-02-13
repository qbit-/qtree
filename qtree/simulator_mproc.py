import numpy as np
from multiprocessing import Pool
from loguru import logger as log

import qtree.utils as utils
import qtree.np_framework as npfr
import qtree.graph_model as gm
import qtree.optimizer as opt
import qtree.operators as ops
from qtree.simulator import get_amplitudes_from_cirq
from qtree.simulator import prepare_parallel_evaluation_np

## Multiprocessing
def work(rank, comm_size, vars_parallel, slice_dict, 
         buckets, data_dict, free_bra_vars):
    # Loop over all amplitudes
    for parallel_slice_dict in utils.slice_values_generator(
        vars_parallel, rank, comm_size):
        # main computation loop. Populate respective slices
        # and do contraction
        log.info('par slice {} :{}', rank, parallel_slice_dict)
        slice_dict.update(parallel_slice_dict)

        sliced_buckets = npfr.get_sliced_np_buckets(
            buckets, data_dict, slice_dict)

        result = opt.bucket_elimination(
            sliced_buckets, npfr.process_bucket_np,
            n_var_nosum=len(free_bra_vars))

        amplitudes = result.data.flatten()
        return amplitudes


def eval_circuit_np_parallel_mproc(filename, initial_state=0):
    """
    Evaluate quantum circuit using MPI to parallelize
    over some of the variables.
    """

    # number of variables to split by parallelization
    # this should be adjusted by the algorithm from memory/cpu
    # requirements
    n_var_parallel = 1

    # TODO: move this
    def filter_bra_vars(bra_vars, free_qubits):
        # TODO: move this
        def print_free_vars_info(free_bra_vars, free_qubits):
            if len(free_bra_vars) > 0:
                print('Evaluate subsets of amplitudes over qubits:')
                print(free_qubits)
                print('Free variables in the resulting expression:')
                print(free_bra_vars)
        # Collect free qubit variables
        free_bra_vars = []
        for ii in free_qubits:
            try:
                free_bra_vars.append(bra_vars[ii])
            except IndexError:
                pass
        bra_vars = [var for var in bra_vars if var not in free_bra_vars]
        print_free_vars_info(free_bra_vars, free_qubits)
        return bra_vars, free_bra_vars

    ## 1. Prepare graph with fixed vars
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    free_qubits = list(range(n_qubits))
    #free_qubits = []
    bra_vars, free_bra_vars = filter_bra_vars(bra_vars, free_qubits)
    graph_initial = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)


    graph_initial = gm.make_clique_on(graph_initial, free_bra_vars)
    ## 2. Parallelize graph
    # find a reduced graph
    ### Should we reduce the graph with clique or make clique on reduced?
    vars_parallel, graph_reduced = gm.split_graph_by_metric_greedy(
    #vars_parallel, graph_reduced = gm.split_graph_random(
        graph_initial, n_var_parallel,
        forbidden_nodes=free_bra_vars,
        metric_fn=gm.get_node_by_mem_reduction)
    log.info('Vars parallel: {}', vars_parallel)

    graph = graph_reduced

    ## 3. Get peo
    peo_initial, treewidth = gm.get_peo(graph)
    log.info('Initial peo: {}', peo_initial)
    # transform peo so free_bra_vars are at the end
    peo = gm.get_equivalent_peo(graph, peo_initial, free_bra_vars)
    #peo = peo[5:] + peo[:5]

    peo =  ket_vars + bra_vars + vars_parallel + peo
    log.info('Final peo: {}', peo)

    ## 4. Prepare vars and buckets
    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    perm_buckets, perm_dict = opt.reorder_buckets(buckets, peo)
    perm_graph, _ = gm.relabel_graph_nodes(
        graph, perm_dict)
    ## *

    #log.info('Perm dict {}',perm_dict)
    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)
    vars_parallel = sorted([perm_dict[idx] for idx in vars_parallel],
                           key=str)

    # Construct slice dictionary for initial state
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)
    target_state = 0
    slice_dict.update(utils.slice_from_bits(target_state, bra_vars))
    slice_dict.update({var: slice(None) for var in free_bra_vars})

    comm_size = 64
    pool = Pool(comm_size)
    log.info(f"Goin' wild with {comm_size} processes, slice: {slice_dict}")

    results = []
    for rank in range(comm_size):
        args = (rank, comm_size, vars_parallel, slice_dict,
                buckets, data_dict, free_bra_vars)
        res = pool.apply_async(work, args)
        results.append(res)

    ampstot = [x.get() for x in results]
    ampstot = [x for x in ampstot if x is not None]
    log.info(f'Got {len(ampstot)} results')

    # TODO: dont be ridiculous, use concat and np.sum
    amplitudes = ampstot[0]
    print(amplitudes.shape)
    for amp in ampstot[1:]:
        print(amp)
        amplitudes +=  amp

    ## 6. Aftermath
    amplitudes_reference = get_amplitudes_from_cirq(filename)
    # Get a slice as we do not need full amplitude
    bra_slices = {var: slice_dict[var] for var in slice_dict
                  if var.name.startswith('o')}

    # sort slice in the big endian order for Cirq
    computed_subtensor = [slice_dict[var]
                          for var in sorted(bra_slices, key=str)]

    slice_of_amplitudes = amplitudes_reference.reshape(
        [2]*n_qubits)[tuple(computed_subtensor)]
    slice_of_amplitudes = slice_of_amplitudes.flatten()
    ##*
    amplitudes_reference = slice_of_amplitudes

    ## Finalize
    print('Result:')
    print(np.round(np.array(amplitudes), 3))
    print('Reference:')
    print(np.round(amplitudes_reference, 3))
    print('Max difference:')
    print(np.max(np.array(amplitudes)
                 - np.array(amplitudes_reference)))

if __name__ == '__main__':
    eval_circuit_np_parallel_mpi('inst_2x2_7_0.txt')
