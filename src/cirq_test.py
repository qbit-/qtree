"""
Test programs to demonstrate various use cases of the
Qtree quantum circuit simulator. Functions in this file
can be used as main functions in the final simulator program
"""
import numpy as np
import tensorflow as tf
import cirq

import src.operators as ops
import src.optimizer as opt
import src.graph_model as gm
import src.np_framework as npfr
import src.tf_framework as tffr
import src.utils as utils

from mpi4py import MPI
from src.quickbb_api import gen_cnf, run_quickbb

QUICKBB_COMMAND = './quickbb/run_quickbb_64.sh'


def get_amplitudes_from_cirq(filename):
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
    simulator = cirq.google.XmonSimulator()

    result = simulator.simulate(cirq_circuit)
    print("Simulation completed\n")

    return result.final_state


def get_optimal_graphical_model(
        filename,
        quickbb_command=QUICKBB_COMMAND):
    """
    Builds a graphical model to contract a circuit in ``filename``
    and finds its tree decomposition
    """
    # filename = 'inst_2x2_1_1.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)

    buckets, graph = opt.circ2buckets(circuit)

    cnffile = 'quickbb.cnf'
    gen_cnf(cnffile, graph)
    run_quickbb(cnffile, quickbb_command)


def eval_circuit(filename, quickbb_command=QUICKBB_COMMAND):
    """
    Loads circuit from file and evaluates all amplitudes
    using the bucket elimination algorithm (with tensorflow tensors).
    Same amplitudes are evaluated with Cirq for comparison.
    """
    # filename = 'inst_4x4_11_2.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)

    # Convert circuit to buckets
    buckets, graph = opt.circ2buckets(circuit)

    # Run quickbb
    if graph.number_of_edges() > 1:  # only if not elementary cliques
        peo, max_mem = gm.get_peo(graph)
        perm_buckets = opt.transform_buckets(buckets, peo)
    else:
        print('QuickBB skipped')
        perm_buckets = buckets

    tf_buckets, placeholder_dict = tffr.get_tf_buckets(perm_buckets, n_qubits)
    comput_graph = opt.bucket_elimination(
        tf_buckets, tffr.process_bucket_tf)

    amplitudes = []
    for target_state in range(2**n_qubits):
        feed_dict = tffr.assign_placeholder_values(
            placeholder_dict,
            target_state, n_qubits)
        amplitude = tffr.run_tf_session(comput_graph, feed_dict)
        amplitudes.append(amplitude)

    amplitudes_reference = get_amplitudes_from_cirq(filename)
    print('Result:')
    print(np.round(np.array(amplitudes), 3))
    print('Reference:')
    print(np.round(amplitudes_reference, 3))


def prepare_parallel_evaluation(filename, n_var_parallel):
    """
    Prepares for parallel evaluation of the quantum circuit.
    Some of the variables in the circuit are parallelized over.
    Symbolic bucket elimination is performed with tensorflow and
    the resulting computation graph (as GraphDef) and other
    supporting information is returned
    """
    # filename = 'inst_2x2_7_0.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)

    # Prepare graphical model
    buckets, graph = opt.circ2buckets(circuit)

    # Run quickBB and get contraction order
    (peo, max_mem,
     idx_parallel, reduced_graph) = gm.get_peo_parallel_by_metric(
         graph, n_var_parallel)

    # Permute buckets to the order of optimal contraction
    perm_buckets = opt.transform_buckets(
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


def eval_circuit_parallel_mpi(filename):
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
        env = prepare_parallel_evaluation(filename, n_var_parallel)
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


def eval_circuit_np(filename, quickbb_command=QUICKBB_COMMAND):
    """
    Loads circuit from file and evaluates all amplitudes
    using the bucket elimination algorithm (with Numpy tensors).
    Same amplitudes are evaluated with Cirq for comparison.
    """
    # filename = 'inst_4x4_11_2.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)

    # Convert circuit to buckets
    buckets, graph = opt.circ2buckets(circuit)

    # Run quickbb
    if graph.number_of_edges() > 1:  # only if not elementary cliques
        peo, max_mem = gm.get_peo(graph)
        perm_buckets = opt.transform_buckets(buckets, peo)
    else:
        print('QuickBB skipped')
        perm_buckets = buckets

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


def prepare_parallel_evaluation_np(filename, n_var_parallel):
    """
    Prepares for parallel evaluation of the quantum circuit.
    Some of the variables in the circuit are parallelized over.
    Unsliced Numpy buckets in the optimal order of elimination
    are returned
    """
    # filename = 'inst_2x2_7_0.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)

    # Prepare graphical model
    buckets, graph = opt.circ2buckets(circuit)

    # Run quickBB and get contraction order
    (peo, max_mem,
     idx_parallel, reduced_graph) = gm.get_peo_parallel_by_metric(
         graph, n_var_parallel)

    # Permute buckets to the order of optimal contraction
    perm_buckets = opt.transform_buckets(
        buckets, peo + idx_parallel)

    environment = dict(
        n_qubits=n_qubits,
        idx_parallel=idx_parallel,
        buckets=perm_buckets
    )

    return environment


def eval_circuit_np_parallel_mpi(filename):
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

    # Loop over all amplitudes
    amplitudes = []
    for target_state in range(2**n_qubits):
        np_buckets = npfr.get_np_buckets(
            buckets, n_qubits, target_state)

        # main computation loop. Populate respective slices
        # and do contraction

        amplitude = 0
        for slice_dict in utils.slice_values_generator(
                comm_size, rank, idx_parallel):
            sliced_buckets = npfr.slice_np_buckets(
                np_buckets, slice_dict, idx_parallel)
            amplitude += opt.bucket_elimination(
                sliced_buckets, npfr.process_bucket_np)

        amplitude = comm.reduce(amplitude, op=MPI.SUM, root=0)
        amplitudes.append(amplitude)

    if rank == 0:
        amplitudes_reference = get_amplitudes_from_cirq(filename)
        print('Result:')
        print(np.round(np.array(amplitudes), 3))
        print('Reference:')
        print(np.round(amplitudes_reference, 3))


def eval_contraction_cost(filename, quickbb_command=QUICKBB_COMMAND):
    """
    Loads circuit from file, evaluates contraction cost
    with and without optimization
    """
    # load circuit
    n_qubits, circuit = ops.read_circuit_file(filename)

    # get contraction graph (node order is arbitrary)
    buckets, _ = opt.circ2buckets(circuit)
    graph_raw = opt.buckets2graph(buckets)

    # estimate cost
    mem_cost, _ = gm.cost_estimator(graph_raw)
    mem_raw = sum(mem_cost)

    # optimize node order
    peo, max_mem = gm.get_peo(graph_raw)

    # get cost for reordered graph
    graph, label_dict = gm.relabel_graph_nodes(
        graph_raw, dict(zip(range(1, len(peo)+1), peo))
    )
    mem_cost, _ = gm.cost_estimator(graph)
    mem_opt = sum(mem_cost)

    # split graph and relabel in optimized way
    n_var_parallel = 3
    peo, _, _, reduced_graph = gm.get_peo_parallel_by_metric(
        graph_raw, n_var_parallel)
    graph_parallel, label_dict = gm.relabel_graph_nodes(
        reduced_graph, dict(zip(range(1, len(peo) + 1), peo))
    )

    mem_cost, _ = gm.cost_estimator(graph_parallel)
    mem_par = sum(mem_cost)

    print('Memory (in doubles):\n raw: {} optimized: {}'.format(
        mem_raw, mem_opt))
    print(' parallel:\n  node: {} total: {} n_tasks: {}'.format(
        mem_par, mem_par*2**(n_var_parallel),
        2**(n_var_parallel)
    ))


if __name__ == "__main__":
    eval_circuit('inst_2x2_7_0.txt')
    eval_circuit_np('inst_2x2_7_0.txt')
    eval_circuit_parallel_mpi('inst_2x2_7_0.txt')
    eval_circuit_np_parallel_mpi('inst_2x2_7_0.txt')
    eval_contraction_cost('inst_2x2_7_0.txt')
