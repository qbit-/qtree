from src.operators import *

import sys
import re
import numpy as np
import logging
log = logging.getLogger('qtree')

OP = qOperation()

def read_circuit_file(filename, max_depth=None):
    log.info("reading file {}".format(filename))
    circuit = []
    circuit_layer = []

    with open(filename, "r") as fp:
        qubit_count = int(fp.readline())
        log.info("There are {:d} qubits in circuit".format(qubit_count))
        n_ignored_layers = 0
        current_layer = 0

        for line in fp:
            m = re.search(r'(?P<layer>[0-9]+) (?=[a-z])', line)
            if m is None:
                raise Exception("file format error at line {}".format(idx))
            # Read circuit layer by layer
            layer_num = int(m.group('layer'))

            if max_depth is not None and layer_num > max_depth:
                n_ignored_layers = layer_num - max_depth
                continue

            if layer_num > current_layer:
                circuit.append(circuit_layer)
                circuit_layer = []
                current_layer = layer_num

            op_str = line[m.end():]
            op = OP.factory(op_str)
            circuit_layer.append(op)

        circuit.append(circuit_layer) # last layer

        if n_ignored_layers > 0:
            log.info("Ignored {} layers".format(n_ignored_layers))

    return qubit_count, circuit


def get_amplitude_from_cirq(filename, target_state_str):
    #filename = 'inst_4x4_10_0.txt'
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

    # target_state_str =  '1010010110100101'
    # reverse the string for index counting
    target_state = [int(i) for i in target_state_str[::-1] ]

    if len(target_state)!=n_qubits:
        raise Exception('target state length is not equal to qbit count')

    amp_idx = np.ravel_multi_index(target_state, [2,]*len(target_state))
    target_amp  = result.final_state[amp_idx]

    print(f'Amplitude of {target_state_str} (index {amp_idx}) is {target_amp}')
    return target_amp, result.final_state


def get_decomposed_graphical_model(filename):
    filename = 'inst_2x2_7_0.txt'
    n_qubits, circuit = read_circuit_file(filename)
    side_length = int(np.sqrt(n_qubits))

    graph = circ2graph(circuit)
    cnffile = 'quickbb.cnf'
    gen_cnf(cnffile, graph)
    #run_quickbb(cnffile)

def test_gates():
    test_cirquits = [
        [X_1_2(0)],
        [Y_1_2(0)],
        [T(0)],
        [X(0)],
        [Y(0)]
    ]
    for circuit in test_cirquits:
        cirq_circuit = cirq.Circuit()
        side_length = 1
        cirq_circuit.append(op.to_cirq_2d_circ_op(side_length) for op in circuit)

        print("Testing circuit "+str(circuit))
        print("Cirq"+str(cirq_circuit))
        simulator = cirq.google.XmonSimulator()

        result = simulator.simulate(cirq_circuit)
        print( result.final_state.round(4))
        print()

if __name__ == "__main__":
    #get_decomposed_graphical_model(sys.argv[1])
    test_gates()
