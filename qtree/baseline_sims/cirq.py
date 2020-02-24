import cirq
import numpy as np

from qtree import log
import qtree.operators as ops


def parse_circuit_2d(n_qubits: int, circuit: list):
    """
    Convert list of layers of gates into cirq circuit
    with qubits on grid (`cirq.GridQubit`)
    """
    side_length = int(np.sqrt(n_qubits))
    cirq_circuit = cirq.Circuit()
    for layer in circuit:
        cirq_circuit.append(op.to_cirq_2d_circ_op(side_length) for op in layer)

    return cirq_circuit


def simulate_state_from_file(filename, initial_state=0):
    """
    Calculates amplitudes for a circuit in file filename using Cirq
    """
    n_qubits, circuit = ops.read_circuit_file(filename)
    cirq_circuit = parse_circuit_2d(n_qubits, circuit)

    print("Circuit:")
    print(cirq_circuit)
    simulator = cirq.Simulator()
    log.info(f"Starting Cirq simulation of {n_qubits} qubits and {len(circuit)} layers")

    result = simulator.simulate(cirq_circuit, initial_state=initial_state)
    log.info("Cirq simulation completed\n")

    # Cirq for some reason computes all amplitudes with phase -1j
    return result.final_state
