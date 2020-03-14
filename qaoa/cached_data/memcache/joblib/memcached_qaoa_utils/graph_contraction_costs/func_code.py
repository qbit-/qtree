# first line: 19
def graph_contraction_costs(size, peo, p=1, type='grid', **kw):
    graph_old, N = memory.cache(qaoa_expr_graph)(size, p, type=type, **kw)
    peo, nghs = memory.cache(neigh_peo)(size, p, type=type, **kw)
    graph, _ = utils.reorder_graph(graph_old, peo)
    costs = qtree.graph_model.cost_estimator(graph)
    return costs
