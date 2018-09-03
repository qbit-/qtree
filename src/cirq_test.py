"""
Test programs to demonstrate various use cases of the
Qtree quantum circuit simulator.
"""
import src.operators as ops
import cirq
import src.optimizer as opt
from src.quickbb_api import gen_cnf, run_quickbb
import src.graph_model as gm
import numpy as np
from mpi4py import MPI
import tensorflow as tf


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
        quickbb_command='./quickbb/run_quickbb_64.sh'):
    """
    Builds a graphical model to contract a circuit in ``filename``
    and finds its tree decomposition
    """
    # filename = 'inst_2x2_1_1.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)

    graph, buckets = opt.circ2buckets(circuit)
    cnffile = 'quickbb.cnf'
    gen_cnf(cnffile, graph)
    run_quickbb(cnffile, quickbb_command)


def eval_circuit(filename, quickbb_command='./quickbb/run_quickbb_64.sh'):
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

    tf_buckets, placeholder_dict = opt.get_tf_buckets(perm_buckets, n_qubits)
    comput_graph = opt.bucket_elimination(
        tf_buckets, opt.process_bucket_tf)

    amplitudes = []
    for target_state in range(2**n_qubits):
        feed_dict = opt.assign_placeholder_values(
            placeholder_dict,
            target_state, n_qubits)
        amplitude = opt.run_tf_session(comput_graph, feed_dict)
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
     idx_parallel, reduced_graph) = gm.get_peo_parallel_random(
         graph, n_var_parallel)

    # Permute buckets to the order of optimal contraction
    perm_buckets = opt.transform_buckets(
        buckets, peo + idx_parallel)

    # Transform tensor labels in buckets to tensorflow placeholders
    # Reset Tensorflow graph as it may store all tensors ever used before
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

    environment = dict(
        n_qubits=n_qubits,
        idx_parallel=idx_parallel,
        input_names=list(pdict_sliced.keys()),
        tf_graph_def=tf.get_default_graph().as_graph_def()
    )

    return environment


def extract_placeholder_dict(tf_graph, variable_names):
    """
    Extract placeholders from the tensorflow computation Graph.

    Returns
    -------
    pdict : dict
        List containing {label : tensorflow placeholder} pairs
    """
    return {
        name: tf_graph.get_tensor_by_name(name + ':0') for
        name in variable_names
    }


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
    placeholder_dict = extract_placeholder_dict(
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
        feed_dict = opt.assign_placeholder_values(
            placeholder_dict,
            target_state, n_qubits)

        # main computation loop. Populate respective slices
        # and do contraction

        amplitude = 0
        for slice_dict in opt.slice_values_generator(
                comm_size, rank, idx_parallel):
            parallel_vars_feed = {
                placeholder_dict[key]: val for key, val
                in slice_dict.items()}

            feed_dict.update(parallel_vars_feed)
            amplitude += opt.run_tf_session(result, feed_dict)

        amplitude = comm.reduce(amplitude, op=MPI.SUM, root=0)
        amplitudes.append(amplitude)

    if rank == 0:
        amplitudes_reference = get_amplitudes_from_cirq(filename)
        print('Result:')
        print(np.round(np.array(amplitudes), 3))    
        print('Reference:')
        print(np.round(amplitudes_reference, 3))


def eval_circuit_np(filename, quickbb_command='./quickbb/run_quickbb_64.sh'):
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
        np_buckets = opt.get_np_buckets(
            perm_buckets, n_qubits, target_state)
        amplitude = opt.bucket_elimination(np_buckets, backend='numpy')
        amplitudes.append(amplitude)

    amplitudes_reference = get_amplitudes_from_cirq(filename)
    print('Result:')
    print(np.round(np.array(amplitudes), 3))    
    print('Reference:')
    print(np.round(amplitudes_reference, 3))


if __name__ == "__main__":
    eval_circuit('test_circuits/inst/cz_v2/4x4/inst_4x4_10_2.txt')
    # eval_circuit_parallel_mpi('inst_4x4_11_2.txt')
