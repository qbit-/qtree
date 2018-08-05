import src.operators as ops
import cirq
import src.optimizer as opt
from src.quickbb_api import gen_cnf, run_quickbb
from src.graph_model import get_peo
import sys
import re
import numpy as np


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
    filename = 'inst_2x2_7_0.txt'
    n_qubits, circuit = ops.read_circuit_file(filename)
    side_length = int(np.sqrt(n_qubits))

    # Run quickbb
    buckets, graph = opt.circ2buckets(circuit)

    if graph.number_of_edges() > 1: #only if not elementary cliques 
        peo, max_mem = get_peo(graph)
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


if __name__ == "__main__":
    contract_with_tensorflow('inst_2x2_7_0.txt')
