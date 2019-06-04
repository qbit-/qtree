"""
Test programs to demonstrate various use cases of the
Qtree quantum circuit simulator. Functions in this file
can be used as main functions in the final simulator program
"""
import numpy as np
import tensorflow as tf
import cirq
import random


import src.operators as ops
import src.optimizer as opt
import src.graph_model as gm
import src.np_framework as npfr
import src.tf_framework as tffr
import src.utils as utils

from mpi4py import MPI
from src.quickbb_api import gen_cnf, run_quickbb

QUICKBB_COMMAND = './quickbb/run_quickbb_64.sh'


def get_amplitudes_from_cirq(filename, initial_state=0):
    """
    Calculates amplitudes for a circuit in file filename using Cirq
    """
    # filename = 'inst_2x2_1_0.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)
    side_length = int(np.sqrt(n_qubits))

    cirq_circuit = cirq.Circuit()

    for layer in circuit:
        cirq_circuit.append(op.to_cirq_2d_circ_op(side_length) for op in layer)

    print("Circuit:")
    print(cirq_circuit)
    simulator = cirq.Simulator()

    result = simulator.simulate(cirq_circuit, initial_state=initial_state)
    print("Simulation completed\n")

    # Cirq for some reason computes all amplitudes with phase -1j
    return result.final_state


def get_optimal_graphical_model(
        filename,
        quickbb_command=QUICKBB_COMMAND):
    """
    Builds a graphical model to contract a circuit in ``filename``
    and finds its tree decomposition
    """
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, free_vars, data_dict = opt.circ2buckets(circuit)
    graph = opt.buckets2graph(buckets)
    peo, tw = gm.get_peo(graph)
    graph_optimal, label_dict = gm.relabel_graph_nodes(graph, dict(zip(
        range(graph.number_of_nodes()), peo)))
    return graph_optimal


def eval_circuit_tf(filename, initial_state=0,
                    quickbb_command=QUICKBB_COMMAND):
    """
    Loads circuit from file and evaluates all amplitudes
    using the bucket elimination algorithm (with tensorflow tensors).
    Same amplitudes are evaluated with Cirq for comparison.
    """
    # Convert circuit to buckets
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, free_vars, data_dict = opt.circ2buckets(n_qubits, circuit)

    graph = opt.buckets2graph(buckets)

    # Run quickbb
    if graph.number_of_edges() > 1:  # only if not elementary cliques
        peo, treewidth = gm.get_peo(graph)
        perm_buckets = opt.reorder_buckets(buckets, peo)
    else:
        print('QuickBB skipped')
        perm_buckets = buckets

    tf_buckets, placeholder_dict = tffr.get_tf_buckets(perm_buckets)
    result = opt.bucket_elimination(
        tf_buckets, tffr.process_bucket_tf)
    comput_graph = result.data

    amplitudes = []
    for target_state in range(2**n_qubits):
        feed_dict = tffr.assign_placeholder_values(
            placeholder_dict, data_dict,
            initial_state, target_state, n_qubits)
        amplitude = tffr.run_tf_session(comput_graph, feed_dict)
        amplitudes.append(amplitude)

    amplitudes_reference = get_amplitudes_from_cirq(filename,
                                                    initial_state)
    print('Result:')
    print(np.round(np.array(amplitudes), 3))
    print('Reference:')
    print(np.round(amplitudes_reference, 3))
    print('Max difference:')
    print(np.max(np.abs(amplitudes - np.array(amplitudes_reference))))


def prepare_parallel_evaluation_tf(filename, n_var_parallel):
    """
    Prepares for parallel evaluation of the quantum circuit.
    Some of the variables in the circuit are parallelized over.
    Symbolic bucket elimination is performed with tensorflow and
    the resulting computation graph (as GraphDef) and other
    supporting information is returned
    """
    # Prepare graphical model
    n_qubits, buckets, free_vars = opt.read_buckets(filename)
    graph = opt.buckets2graph(buckets)

    # Run quickBB and get contraction order
    idx_parallel, reduced_graph = gm.split_graph_by_metric(
         graph, n_var_parallel)
    peo, treewidth = gm.get_peo(reduced_graph)

    # Permute buckets to the order of optimal contraction
    perm_buckets = opt.reorder_buckets(
        buckets, peo + idx_parallel)

    # Transform tensor labels in buckets to tensorflow placeholders
    # Reset Tensorflow graph as it may store all tensors ever used before
    tf.reset_default_graph()
    tf_buckets, placeholder_dict = tffr.get_tf_buckets(
        perm_buckets, n_qubits)

    # Apply slicing as we parallelize over some variables
    sliced_tf_buckets, pdict_sliced = tffr.slice_tf_buckets(
        tf_buckets, placeholder_dict, idx_parallel)

    # Do symbolic computation of the result
    result = tf.identity(
        opt.bucket_elimination(
            sliced_tf_buckets, tffr.process_bucket_tf),
        name='result'
    )

    environment = dict(
        n_qubits=n_qubits,
        idx_parallel=idx_parallel,
        input_names=list(pdict_sliced.keys()),
        tf_graph_def=tf.get_default_graph().as_graph_def()
    )

    return environment


def eval_circuit_tf_parallel_mpi(filename):
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
    placeholder_dict = tffr.extract_placeholder_dict(
        tf.get_default_graph(),
        env['input_names']
    )
    result = tf.get_default_graph().get_tensor_by_name('result:0')

    # restore other parts of the environment
    n_qubits = env['n_qubits']
    idx_parallel = env['idx_parallel']

    # Loop over all amplitudes
    amplitudes = []
    for target_state in range(2**n_qubits):
        feed_dict = tffr.assign_placeholder_values(
            placeholder_dict,
            target_state, n_qubits)

        # main computation loop. Populate respective slices
        # and do contraction

        amplitude = 0
        for slice_dict in utils.slice_values_generator(
                comm_size, rank, idx_parallel):
            parallel_vars_feed = {
                placeholder_dict[key]: val for key, val
                in slice_dict.items()}

            feed_dict.update(parallel_vars_feed)
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


def eval_circuit_np(filename, initial_state=0,
                    quickbb_command=QUICKBB_COMMAND):
    """
    Loads circuit from file and evaluates all amplitudes
    using the bucket elimination algorithm (with Numpy tensors).
    Same amplitudes are evaluated with Cirq for comparison.
    """
    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, free_vars, data_dict = opt.circ2buckets(n_qubits, circuit)

    graph = opt.buckets2graph(buckets)

    # Run quickbb
    if graph.number_of_edges() > 1:  # only if not elementary cliques
        peo, treewidth = gm.get_peo(graph)
        perm_buckets = opt.reorder_buckets(buckets, peo)
    else:
        print('QuickBB skipped')
        perm_buckets = buckets

    amplitudes = []
    for target_state in range(2**n_qubits):
        np_buckets = npfr.get_np_buckets(
            perm_buckets, data_dict, initial_state,
            target_state, n_qubits)
        result = opt.bucket_elimination(
            np_buckets, npfr.process_bucket_np)
        amplitudes.append(result.data)

    # Cirq returns the amplitudes in big endian (largest bit first)

    amplitudes_reference = get_amplitudes_from_cirq(filename,
                                                    initial_state)
    print('Result:')
    print(np.round(np.array(amplitudes), 3))
    print('Reference:')
    print(np.round(np.array(amplitudes_reference), 3))
    print('Max difference:')
    print(np.max(np.abs(
        np.array(amplitudes) - np.array(amplitudes_reference))))


def prepare_parallel_evaluation_np(filename, n_var_parallel):
    """
    Prepares for parallel evaluation of the quantum circuit.
    Some of the variables in the circuit are parallelized over.
    Unsliced Numpy buckets in the optimal order of elimination
    are returned
    """
    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, free_vars, data_dict = opt.circ2buckets(n_qubits, circuit)

    graph = opt.buckets2graph(buckets)

    # Run quickBB and get contraction order
    idx_parallel, reduced_graph = gm.split_graph_by_metric(
        graph, n_var_parallel)
    peo, treewidth = gm.get_peo(reduced_graph)

    # Permute buckets to the order of optimal contraction
    perm_buckets = opt.reorder_buckets(
        buckets, peo + idx_parallel)

    environment = dict(
        n_qubits=n_qubits,
        idx_parallel=idx_parallel,
        buckets=perm_buckets,
        data_dict=data_dict
    )

    return environment


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

    # restore buckets
    buckets = env['buckets']

    # restore other parts of the environment
    n_qubits = env['n_qubits']
    idx_parallel = env['idx_parallel']

    # restore data dictionary
    data_dict = env['data_dict']

    # Loop over all amplitudes
    amplitudes = []
    for target_state in range(2**n_qubits):
        np_buckets = npfr.get_np_buckets(
            buckets, data_dict, initial_state, target_state, n_qubits)

        # main computation loop. Populate respective slices
        # and do contraction

        amplitude = 0
        for slice_dict in utils.slice_values_generator(
                comm_size, rank, idx_parallel):
            sliced_buckets = npfr.slice_np_buckets(
                np_buckets, slice_dict, idx_parallel)
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


def eval_contraction_cost(filename, quickbb_command=QUICKBB_COMMAND):
    """
    Loads circuit from file, evaluates contraction cost
    with and without optimization
    """
    # Prepare graphical model
    n_qubits, buckets, free_vars = opt.read_buckets(filename)
    graph_raw = opt.buckets2graph(buckets)

    # estimate cost
    mem_raw, flop_raw = gm.cost_estimator(graph_raw)
    mem_raw_tot = sum(mem_raw)

    # optimize node order
    peo, treewidth = gm.get_peo(graph_raw)

    # get cost for reordered graph
    graph, label_dict = gm.relabel_graph_nodes(
        graph_raw, dict(zip(peo, range(1, len(peo)+1)))
    )
    mem_opt, flop_opt = gm.cost_estimator(graph)
    mem_opt_tot = sum(mem_opt)

    # split graph and relabel in optimized way
    n_var_parallel = 3
    _, reduced_graph = gm.split_graph_by_metric(
        graph_raw, n_var_parallel)
    peo, treewidth = gm.get_peo(reduced_graph)

    graph_parallel, label_dict = gm.relabel_graph_nodes(
        reduced_graph, dict(zip(peo, range(1, len(peo) + 1)))
    )

    mem_par, flop_par = gm.cost_estimator(graph_parallel)
    mem_par_tot = sum(mem_par)

    print('Memory (in doubles):\n raw: {} optimized: {}'.format(
        mem_raw_tot, mem_opt_tot))
    print(' parallel:\n  node: {} total: {} n_tasks: {}'.format(
        mem_par_tot, mem_par_tot*2**(n_var_parallel),
        2**(n_var_parallel)
    ))


def test_graph_reading(filename):
    """
    This function tests direct reading of circuits to graphs.
    It should be noted that graphs can not to be used in place
    of buckets yet, since the information about transpositions
    of tensors (denoted by edges) is not kept during node
    relabelling
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import pprint as pp

    n_qubits, graph = gm.read_graph(filename)

    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets_original, graph_original = opt.circ2buckets(circuit)

    from networkx.algorithms import isomorphism
    GM = isomorphism.GraphMatcher(graph, graph_original)

    print('Isomorphic? : {}'.format(GM.is_isomorphic()))
    graph = nx.relabel_nodes(graph, GM.mapping, copy=True)

    gm.draw_graph(graph, 'new_graph.png')
    gm.draw_graph(graph_original, 'orig_graph.png')

    buckets = opt.graph2buckets(graph)
    buckets_from_original = opt.graph2buckets(graph_original)

    print('Original buckets')
    pp.pprint(buckets_original)
    print('Buckets from graph')
    pp.pprint(buckets_from_original)
    print('New buckets')
    pp.pprint(buckets)


def test_bucket_reading(filename):
    """
    This function tests direct reading of circuits to buckets.
    """
    n_qubits, buckets, free_vars = opt.read_buckets(filename)
    graph = opt.buckets2graph(buckets)

    peo, treewidth = gm.get_peo(graph)
    perm_buckets = opt.reorder_buckets(buckets, peo)

    amplitudes = []
    for target_state in range(2**n_qubits):
        np_buckets = npfr.get_np_buckets(
            perm_buckets, n_qubits, target_state)
        amplitude = opt.bucket_elimination(
            np_buckets, npfr.process_bucket_np)
        amplitudes.append(amplitude)

    amplitudes_reference = get_amplitudes_from_cirq(filename)
    print('Result:')
    print(np.round(np.array(amplitudes), 3))
    print('Reference:')
    print(np.round(amplitudes_reference, 3))
    print('Maximal difference:')
    print(np.max(np.array(amplitudes)
                 - np.array(amplitudes_reference)))


def what_is_terminal_tensor():
    """
    This functions shows what a terminal tensor (e.g. the one
    we encounter when trying to calculate several amplitudes at a time)
    should be. Here 4 qubits are considered. This tensor
    is made of ones.
    """
    import itertools

    def kron_list(operands):
        res = operands[0].flatten()
        shapes = [res.shape[0]]

        for operand in operands[1:]:
            shapes.append(operand.flatten().shape[0])
            res = np.kron(res, operand.flatten())
        return np.reshape(res, shapes)

    u = np.array([0, 1])
    d = np.array([1, 0])

    a = np.zeros((2, 2, 2, 2))

    for prod in itertools.product([u, d], repeat=4):
        a += kron_list(prod)
    print(a)


def eval_circuit_multiamp_np(
        filename, quickbb_command=QUICKBB_COMMAND):
    """
    Loads circuit from file and evaluates
    multiple amplitudes at once using np framework
    """
    # Get the number of qubits
    n_qubits, _, _ = opt.read_buckets(filename)

    # Read buckets with all qubits set to free variables
    free_qubits = [1, 3]
    n_qubits, buckets, free_vars = opt.read_buckets(filename,
                                                    free_qubits)
    if len(free_vars) > 0:
        print('Evaluate subsets of amplitudes over qubits:')
        print(free_qubits)
        print('Free variables in the resulting expression:')
        print(free_vars)

    graph_initial = opt.buckets2graph(buckets)
    graph = gm.make_clique_on(graph_initial, free_vars)

    # Run quickbb
    peo_initial, _ = gm.get_peo(graph)
    treewidth = gm.get_treewidth_from_peo(graph, peo_initial)
    peo = gm.get_equivalent_peo(graph, peo_initial, free_vars)

    # Apply calculated PEO
    perm_buckets = opt.reorder_buckets(buckets, peo)
    perm_graph, _ = gm.relabel_graph_nodes(
        graph, dict(zip(peo, range(1, len(peo)+1))))

    # Finally make numpy buckets and calculate
    np_buckets = npfr.get_np_buckets(
        perm_buckets, n_qubits, 0)
    amplitude = opt.bucket_elimination(
        np_buckets, npfr.process_bucket_np,
        n_var_nosum=len(free_vars))

    # Take reverse of the amplitude
    amplitudes = amplitude.flatten()[::-1]

    # Now calculate the reference
    amplitudes_reference = get_amplitudes_from_cirq(filename)

    # Get a slice as we do not need full amplitude
    computed_slice = []
    for qubit_idx, qubit_val in zip(
            range(n_qubits),
            utils.int_to_bitstring(0, n_qubits)):
        if qubit_idx in free_qubits:
            computed_slice.append(slice(None))
        else:
            computed_slice.append(int(qubit_val))
    slice_of_amplitudes = amplitudes_reference.reshape(
        [2]*n_qubits)[tuple(computed_slice)]
    slice_of_amplitudes = slice_of_amplitudes.flatten()

    print('Result:')
    print(np.round(amplitudes, 3))
    print('Reference:')
    print(np.round(slice_of_amplitudes, 3))
    print('Max difference:')
    print(np.max(np.abs(amplitudes - slice_of_amplitudes)))


if __name__ == "__main__":
    eval_circuit_tf('inst_2x2_7_1.txt', 1)
    eval_circuit_np('inst_2x2_7_1.txt', 1)
    eval_circuit_tf_parallel_mpi('inst_2x2_7_0.txt')
    eval_circuit_np_parallel_mpi('inst_2x2_7_0.txt')
    eval_contraction_cost('inst_2x2_7_0.txt')
    eval_circuit_multiamp_np('inst_2x2_7_0.txt')
