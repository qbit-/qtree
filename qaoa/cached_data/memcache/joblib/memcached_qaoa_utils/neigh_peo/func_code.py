# first line: 14
def neigh_peo(size, p=1, type='grid'):
    graph, N = memory.cache(qaoa_expr_graph)(size, p, type=type)
    peo, nghs = utils.get_locale_peo(graph, utils.n_neighbors)
    return peo, nghs
