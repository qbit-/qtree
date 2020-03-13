import joblib
import sys
sys.path.append('..')
import utils_qaoa as qaoa
import qtree
import utils

memory = joblib.Memory('./cached_data/memcache')

@memory.cache
def qaoa_expr_graph(size, p=1):
    return qaoa.get_test_expr_graph(size, p)

@memory.cache
def neigh_peo(size, p=1):
    graph, N = qaoa_expr_graph(size, p)
    peo, nghs = utils.get_locale_peo(graph, utils.n_neighbors)
    return peo, nghs

@memory.cache
def graph_contraction_costs(size, peo, p=1):
    graph_old, N = qaoa_expr_graph(size, p)
    peo, nghs = neigh_peo(size, p)
    graph, _ = utils.reorder_graph(graph_old, peo)
    costs = qtree.graph_model.cost_estimator(graph)
    return costs

@memory.cache
def contracted_graph(size, peo, idx, p=1):
    graph, N = qaoa_expr_graph(size, p)
    peo, nghs = neigh_peo(size, p)
    for n in peo[:idx]:
        qtree.graph_model.eliminate_node(graph, n)
    return graph
