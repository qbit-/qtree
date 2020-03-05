import qtree

def layer_of_Hadamards(qc,N):
    layer = []
    for q in range(N):
        layer.append(qtree.operators.H(q))
    qc.append(layer)

def get_qaoa_circuit(G, beta, gamma):
    assert(len(beta) == len(gamma))
    p = len(beta) # infering number of QAOA steps from the parameters passed
    N = G.number_of_nodes()
    qc = []
    layer_of_Hadamards(qc, N)
    # second, apply p alternating operators
    for i in range(p):
        qc += get_cost_operator_circuit(G,gamma[i])
        qc += get_mixer_operator_circuit(G,beta[i])
    # finally, do not forget to measure the result!
    return qc

def append_x_term(qc, q1, beta):
    layer = []
    layer.append(qtree.operators.H(q1))
    layer.append(qtree.operators.ZPhase(q1, alpha=2*beta))
    layer.append(qtree.operators.H(q1))
    qc.append(layer)

def get_mixer_operator_circuit(G, beta):
    N = G.number_of_nodes()
    qc = []
    for n in G.nodes():
        append_x_term(qc, n, beta)
    return qc

def append_zz_term(qc, q1, q2, gamma):
    layer = []
    layer.append(qtree.operators.cX(q1, q2))
    layer.append(qtree.operators.ZPhase(q2, alpha=2*gamma))
    layer.append(qtree.operators.cX(q1, q2))
    qc.append(layer)

def get_cost_operator_circuit(G, gamma):
    N = G.number_of_nodes()
    qc = list()
    for i, j in G.edges():
        append_zz_term(qc, i, j, gamma)
    return qc
