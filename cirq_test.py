from src.operators import *
from src.logging import log
from src.optimizer  import *
from src.quickbb_api import gen_cnf, run_quickbb
import sys
import re
import numpy as np
OP = qOperation()


def get_amplitudes_from_cirq(filename):
    #filename = 'inst_2x2_1_0.txt'
    n_qubits, circuit = read_circuit_file(filename)
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

    # target_state_str =  '1010010110100101'
    # reverse the string for index counting
    # multi_idx = [int(i) for i in target_state_str[::-1] ]

    # if len(multi_idx)!=n_qubits:
    #     raise Exception('target state length is not equal to qbit count')

    # lin_idx = np.ravel_multi_index(multi_idx, [2,]*len(multi_idx))
    # target_amp  = result.final_state[lin_idx]

    # print(f'Amplitude of {target_state_str} (index {amp_idx}) is {target_amp}')
    # return target_amp


def get_decomposed_graphical_model(filename):
    filename = 'inst_2x2_7_0.txt'
    n_qubits, circuit = read_circuit_file(filename)
    side_length = int(np.sqrt(n_qubits))

    graph, buckets = circ2buckets(circuit)
    cnffile = 'quickbb.cnf'
    gen_cnf(cnffile, graph)
    run_quickbb(cnffile, './quickbb/run_quickbb_64.sh')


def contract_with_tensorflow(filename):
    filename = 'inst_2x2_1_1.txt'
    n_qubits, circuit = read_circuit_file(filename)
    side_length = int(np.sqrt(n_qubits))

    # Run quickbb
    buckets, graph = circ2buckets(circuit)

    if graph.number_of_edges() > 1: #only if not elementary cliques 
        cnffile = 'quickbb.cnf'
        gen_cnf(cnffile, graph)
        out_bytes = run_quickbb(cnffile, './quickbb/run_quickbb_64.sh')

        # Extract order
        m = re.search(b'(?P<peo>(\d+ )+).*Treewidth=(?P<treewidth>\s\d+)',
                      out_bytes, flags=re.MULTILINE | re.DOTALL )

        peo = np.array([int(ii) for ii in m['peo'].split()])
        treewidth = int(m['treewidth'])

        print(peo)
        print(treewidth)
        
        perm_buckets = transform_buckets(buckets, peo)
    else:
        print('QuickBB skipped')
        perm_buckets = buckets

    tf_buckets, placeholder_dict = get_tf_buckets(perm_buckets, n_qubits)
    comput_graph = bucket_elimination(tf_buckets)

    amplitudes = []
    for target_state in range(2**n_qubits):
        feed_dict = assign_placeholder_values(
            placeholder_dict,
            target_state, n_qubits)
        amplitude = run_tf_session(comput_graph, feed_dict)
        amplitudes.append(amplitude)

    amplitudes_reference = get_amplitudes_from_cirq(filename)
    print('Result:')
    print(np.array(amplitudes))    
    print('Reference:')
    print(amplitudes_reference)


if __name__ == "__main__":
    contract_with_tensorflow(1)
    # amplitudes_reference = get_amplitudes_from_cirq('inst_2x2_1_1.txt')
    # print('Reference:')
    # print(np.round(amplitudes_reference, 3))
