# first line: 10
@memory.cache
def qaoa_expr_graph(size, p=1):
    return qaoa.get_test_expr_graph(size, p)
