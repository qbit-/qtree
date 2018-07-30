from src.operators import *
from src.logging import log
import sys
import re
import numpy as np
OP = qOperation()


def read_test(filename, max_depth=None):
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


def main():
    # filename = sys.argv[1]
    filename = 'inst_4x4_10_0.txt'
    n_qubits, circuit = read_test(filename)
    side_length = int(np.sqrt(n_qubits))
    
    cirq_circuit = cirq.Circuit()

    for layer in circuit:
        cirq_circuit.append(op.to_cirq(side_length) for op in layer)
        
    print("Circuit:")
    print(cirq_circuit)
    simulator = cirq.google.XmonSimulator()

    result = simulator.simulate(cirq_circuit)
    print("Results:")
    print(result)

    print("DONE\n")


if __name__ == "__main__":
    main()


with open('inst_4x4_10_0.txt', 'r+') as fp: 
    line_one = fp.readline()
    print(f'line_1: {line_one}')
    for n, line in enumerate(fp):
        print(f'line {n}: {line}')
