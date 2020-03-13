# first line: 14
@memory.cache
def neigh_peo(size, p=1):
    graph, N = qaoa_expr_graph(size, p)
    peo, nghs = utils.get_locale_peo(graph, utils.n_neighbors)
    return peo, nghs
