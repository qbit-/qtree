import argparse
from src.logging import get_logger
get_logger()

from src.operators import *
from src.optimizer  import *
from src.quickbb_api import *
from cirq_test import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('circuitfile', help='file with circuit')
    parser.add_argument('target_state',
                        help='state x against which amplitude is computed')
    args = parser.parse_args()

    target_amp,state = get_amplitude_from_cirq(
       args.circuitfile, args.target_state)

    n_qubits, circuit = read_circuit_file(args.circuitfile)

    graph , tensors = circ2graph(circuit)

    amp = naive_eliminate(graph,tensors)
    print('cirq',state.round(2))
    log.info('amp of |0> is'+str(amp))
    log.info("from cirq:"+str(target_amp))
    print()
    cnffile = 'quickbb.cnf'
    gen_cnf(cnffile,graph)
    run_quickbb(cnffile)

if __name__=="__main__":
    main()
