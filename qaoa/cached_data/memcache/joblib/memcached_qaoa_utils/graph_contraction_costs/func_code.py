# first line: 19
def graph_contraction_costs(size, peo, p=1, type='grid'):
    graph_old, N = qaoa_expr_graph(size, p, type=type)
    peo, nghs = neigh_peo(size, p, type=type)
    graph, _ = utils.reorder_graph(graph_old, peo)
    costs = qtree.graph_model.cost_estimator(graph)
    return costs
