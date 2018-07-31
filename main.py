import argparse

from src.operators import *
from src.logging import log
from src.optimiser  import *
from src.quickbb_api import *
from cirq_test import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('circuitfile', help='file with circuit')
    parser.add_argument('target_state',
                        help='state x against which amplitude is computed')
    args = parser.parse_args()

    #target_amp = get_amplitude_from_cirq(
    #   args.circuitfile, args.target_state)

    n_qubits, circuit = read_circuit_file(args.circuitfile)

    graph = circ2graph(circuit)
    cnffile = 'quickbb.cnf'
    gen_cnf(cnffile,graph)
    run_quickbb(cnffile)

if __name__=="__main__":
    main()
