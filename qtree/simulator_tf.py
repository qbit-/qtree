import numpy as np
import tensorflow as tf

import qtree.operators as ops
import qtree.optimizer as opt
import qtree.graph_model as gm
import qtree.utils as utils
import qtree.tf_framework as tffr

from qtree.simulator import get_amplitudes_from_cirq

def eval_circuit_tf(filename, initial_state=0):
    """
    Loads circuit from file and evaluates all amplitudes
    using the bucket elimination algorithm (with tensorflow tensors).
    Same amplitudes are evaluated with Cirq for comparison.
    """
    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    graph = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)

    # Run quickbb
    peo, treewidth = gm.get_peo(graph)
    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    peo = ket_vars + bra_vars + peo
    perm_buckets, perm_dict = opt.reorder_buckets(buckets, peo)

    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)

    # Populate slice dict. Only shapes of slices are needed at this stage
    slice_dict = utils.slice_from_bits(initial_state, ket_vars)
    slice_dict.update(utils.slice_from_bits(
        initial_state, bra_vars))

    # create placeholders with proper shapes
    tf_buckets, placeholders_dict = tffr.get_sliced_tf_buckets(
        perm_buckets, slice_dict)

    # build the Tensorflow operation graph
    result = opt.bucket_elimination(
        tf_buckets, tffr.process_bucket_tf)
    comput_graph = result.data

    # prepare static part of the feed_dict
    feed_dict = tffr.assign_tensor_placeholders(
        placeholders_dict, data_dict)

    amplitudes = []
    for target_state in range(2**n_qubits):
        # Now the bounds of slices are needed
        slice_dict.update(
            utils.slice_from_bits(target_state, bra_vars))
        # populate feed dict with slice variables
        feed_dict.update(tffr.assign_variable_placeholders(
            placeholders_dict, slice_dict))

        amplitude = tffr.run_tf_session(comput_graph, feed_dict)
        amplitudes.append(amplitude)

    amplitudes_reference = get_amplitudes_from_cirq(filename,
                                                    initial_state)
    print('Result:')
    print(np.round(np.array(amplitudes), 3))
    print('Reference:')
    print(np.round(amplitudes_reference, 3))
    print('Max difference:')
    print(np.max(np.abs(amplitudes - np.array(amplitudes_reference))))


def prepare_parallel_evaluation_tf(filename, n_var_parallel):
    """
    Prepares for parallel evaluation of the quantum circuit.
    Some of the variables in the circuit are parallelized over.
    Symbolic bucket elimination is performed with tensorflow and
    the resulting computation graph (as GraphDef) and other
    supporting information is returned
    """
    # Prepare graphical model
    n_qubits, circuit = ops.read_circuit_file(filename)
    buckets, data_dict, bra_vars, ket_vars = opt.circ2buckets(
        n_qubits, circuit)

    graph = gm.buckets2graph(
        buckets,
        ignore_variables=bra_vars+ket_vars)

    # find a reduced graph
    vars_parallel, graph_reduced = gm.split_graph_by_metric(
        graph, n_var_parallel,
        metric_fn=gm.get_node_by_mem_reduction)

    # run quickbb once again to get peo and treewidth
    peo, treewidth = gm.get_peo(graph_reduced)

    # place bra and ket variables to beginning, so these variables
    # will be contracted first
    peo = ket_vars + bra_vars + vars_parallel + peo
    perm_buckets, perm_dict = opt.reorder_buckets(buckets, peo)

    # extract bra and ket variables from variable list and sort according
    # to qubit order
    ket_vars = sorted([perm_dict[idx] for idx in ket_vars], key=str)
    bra_vars = sorted([perm_dict[idx] for idx in bra_vars], key=str)
    vars_parallel = sorted([perm_dict[idx] for idx in vars_parallel],
                           key=str)

    # Populate slice dict. Only shapes of slices are needed at this stage
    slice_dict = utils.slice_from_bits(
        0, bra_vars + ket_vars + vars_parallel)

    # create placeholders with proper shapes
    tf.reset_default_graph()
    tf_buckets, placeholders_dict = tffr.get_sliced_tf_buckets(
        perm_buckets, slice_dict)
    # save only placeholder's names as they are not picklable
    picklable_placeholders = {key.name: val for key, val in
                              placeholders_dict.items()}

    # Do symbolic computation of the result
    #result = opt.bucket_elimination(
    #    tf_buckets, tffr.process_bucket_tf)
    #comput_graph = tf.identity(result.data, name='result')

    environment = dict(
        bra_vars=bra_vars,
        ket_vars=ket_vars,
        vars_parallel=vars_parallel,
        tf_graph_def=tf.get_default_graph().as_graph_def(),
        data_dict=data_dict,
        picklable_placeholders=picklable_placeholders
    )

    return environment

if __name__ == "__main__":
    eval_circuit_tf('inst_2x2_7_0.txt')
