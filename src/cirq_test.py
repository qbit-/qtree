import src.operators as ops
import cirq
import src.optimizer as opt
from src.quickbb_api import gen_cnf, run_quickbb
import src.graph_model as gm
import sys
import re
import numpy as np
from mpi4py import MPI
import tensorflow as tf


def get_amplitudes_from_cirq(filename):
    #filename = 'inst_2x2_1_0.txt'
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


def get_decomposed_graphical_model(
        filename,
        quickbb_command='./quickbb/run_quickbb_64.sh'):
    #filename = 'inst_2x2_1_1.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)
    side_length = int(np.sqrt(n_qubits))

    graph, buckets = opt.circ2buckets(circuit)
    cnffile = 'quickbb.cnf'
    gen_cnf(cnffile, graph)
    run_quickbb(cnffile, quickbb_command)


def contract_with_tensorflow(filename, quickbb_command='./quickbb/run_quickbb_64.sh'):
    # filename = 'inst_2x2_7_0.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)
    side_length = int(np.sqrt(n_qubits))

    # Run quickbb
    buckets, graph = opt.circ2buckets(circuit)

    if graph.number_of_edges() > 1: #only if not elementary cliques 
        peo, max_mem = gm.get_peo(graph)
        perm_buckets = opt.transform_buckets(buckets, peo)
    else:
        print('QuickBB skipped')
        perm_buckets = buckets

    tf_buckets, placeholder_dict = opt.get_tf_buckets(perm_buckets, n_qubits)
    comput_graph = opt.bucket_elimination(tf_buckets)

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

    
def prepare_paralell_contraction(filename):
    #filename = 'inst_2x2_7_0.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)
    side_length = int(np.sqrt(n_qubits))

    # Prepare graphical model
    buckets, graph = opt.circ2buckets(circuit)

    # Run quickBB and get contraction order
    peo, max_mem, idx_parallel, reduced_graph = gm.get_peo_parallel_random(graph, 2)

    # Permute buckets to the order of optimal contraction
    perm_buckets = opt.transform_buckets_parallel(
        buckets, peo, idx_parallel)

    # Transform tensor labels in buckets to tensorflow placeholders
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
        tf_graph_def=tf.get_default_graph().as_graph_def(),
        input_names=list(placeholder_dict.keys())
    )
    
    return environment


def extract_placeholder_dict(tf_graph, variable_names):
    return {
        name : tf_graph.get_tensor_by_name(name + ':0') for 
        name in variable_names
    }


def mpi_parallel_contraction(filename):
    """
    Contract quantum circuit using MPI to parallelize
    over some variables
    """
    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank
    status = MPI.Status()

    if rank == 0:
        env = prepare_paralell_contraction(filename)
    else:
        env = None

    env = comm.bcast(env, root=0)
    

    # restore tensorflow graph, extract inputs and outputs
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
        for slice_dict in opt.slice_values_generator(comm_size, rank, idx_parallel):
            parallel_vars_feed = {
                placeholder_dict[key] : val for key, val
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
        

if __name__ == "__main__":
    # contract_with_tensorflow('inst_2x2_7_1.txt')
    mpi_parallel_contraction('inst_2x2_7_0.txt')
