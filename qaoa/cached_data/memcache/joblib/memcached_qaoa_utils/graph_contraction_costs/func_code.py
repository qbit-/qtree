# first line: 20
@memory.cache
def graph_contraction_costs(size, peo, p=1):
    graph_old, N = qaoa_expr_graph(size, p)
    peo, nghs = neigh_peo(size, p)
    graph, _ = utils.reorder_graph(graph_old, peo)
    costs = qtree.graph_model.cost_estimator(graph)
    return costs
