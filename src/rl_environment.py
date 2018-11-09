"""
This module is for working with reinforcement learning agents
for computing the tree decomposition of the expression graphs
"""
import numpy as np
import networkx as nx
import src.graph_model as gm
import copy

MAX_STATE_SIZE = 10


def sparse_graph_adjacency(G, max_size, node_to_idx, weight='weight'):
    """Return the graph adjacency matrix as a SciPy sparse matrix.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the adjacency matrix.

    max_size : int
        Matrix size. May be larger than the number of nodes. Has to
        be compatible with the node_to_idx mapping.

    node_to_idx : dict
        The elements of the adjacency matrix
        are placed in the node_to_idx position

    Returns
    -------
    M : scipy.sparse
        Zero padded adjacency matrix
    """
    from scipy import sparse

    nodelist = list(G)

    if not set(nodelist).issubset(node_to_idx):
        msg = "`nodelist` is not a subset of the `node_to_idx` dictionary."
        raise nx.NetworkXError(msg)

    index = {node: node_to_idx[node] for node in nodelist}
    coefficients = zip(*((index[u], index[v], d.get(weight, 1))
                         for u, v, d in G.edges(nodelist, data=True)
                         if u in index and v in index))
    try:
        row, col, data = coefficients
    except ValueError:
        # there is no edge in the subgraph
        row, col, data = [], [], []

    # symmetrize matrix
    d = data + data
    r = row + col
    c = col + row
    # selfloop entries get double counted when symmetrizing
    # so we subtract the data on the diagonal
    selfloops = list(nx.selfloop_edges(G, data=True))
    if selfloops:
        diag_index, diag_data = zip(*((index[u], -d.get(weight, 1))
                                      for u, v, d in selfloops
                                      if u in index and v in index))
        d += diag_data
        r += diag_index
        c += diag_index
    M = sparse.coo_matrix((d, (r, c)), shape=(max_size, max_size))
    return M


def print_int_matrix(matrix):
    """
    Prints integer matrix in a readable form
    """
    for row in environment.state:
        line = ' '.join(f'{e:d}' if e != 0 else '-' for e in row)
        print(line)

    
class Environment:
    """
    Creates an environment to train the agents
    """
    def __init__(self, filename):
        """
        Creates an environment for the model from file

        Parameters
        ----------
        filename : str
               file to load
        """
        n_qubits, initial_graph = gm.read_graph(filename)
        if initial_graph.number_of_nodes() > MAX_STATE_SIZE:
            raise ValueError(
                f'Graph is larger than the maximal state size:' +
                f' {MAX_STATE_SIZE}')
        self.initial_graph = initial_graph

        self.reset()

    def reset(self):
        """
        Resets the state of the environment. The graph is
        randomly permutted and a new adjacency matrix is generated
        """

        n_nodes = self.initial_graph.number_of_nodes()

        graph_indices = np.random.permutation(range(1, n_nodes+1))
        entry_indices = np.random.choice(MAX_STATE_SIZE, n_nodes,
                                         replace=False)
        node_to_idx = dict(zip(graph_indices, entry_indices))
        idx_to_node = dict(zip(entry_indices, graph_indices))

        graph = copy.deepcopy(self.initial_graph)
        adj_matrix = np.asarray(
            sparse_graph_adjacency(
                graph, MAX_STATE_SIZE, node_to_idx).todense()
            )
        # state = adj_matrix[np.tril_indices_from(adj_matrix)]
        state = adj_matrix

        # Store state and useful mappings
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.graph = graph
        self.state = state

    def step(self, index):
        """
        Takes 1 step in the graph elimination environment

        Parameters
        ----------
        indes : int
              index in the state matrix to eliminate.
        """
        node = self.idx_to_node[index]
        
        # Calculate cost function
        memory, flops = gm.get_cost_by_node(self.graph, node)

        # Update state
        gm.eliminate_node(self.graph, node)
        complete = self.graph.number_of_nodes() == 0

        adj_matrix = np.asarray(
            sparse_graph_adjacency(self.graph, MAX_STATE_SIZE,
                                   self.node_to_idx).todense()
        )
        # self.state = adj_matrix[np.tril_indices_from(adj_matrix)]
        self.state = adj_matrix

        return flops, complete


if __name__ == '__main__':
    environment = Environment('inst_2x2_7_0.txt')
    environment.reset()

    costs = []
    steps = []
    complete = False
    while not complete:
        print_int_matrix(environment.state)
        print()

        row, col = np.nonzero(environment.state)
        cost, complete = environment.step(row[0])

        steps.append(environment.idx_to_node[row[0]])
        costs.append(cost)

    print(' Strategy\n Node | Cost:')
    print('-'*24)
    print('\n'.join('{:5} | {:5}'.format(step, cost)
                    for step, cost in zip(steps, costs)))
    print('-'*24)
    print('Total cost: {}'.format(sum(costs)))
