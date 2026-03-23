from math import ceil

import networkx as nx
import numpy as np
import scipy as sp


def generate_planar_graphs(num_graphs, min_size, max_size=None, seed=0):
    """Generate planar graphs using Delauney triangulation."""
    rng = np.random.default_rng(seed)
    graphs: list[nx.Graph] = []

    max_size = max_size or min_size

    for _ in range(num_graphs):
        n = rng.integers(min_size, max_size, endpoint=True)
        points = rng.random((n, 2))
        tri = sp.spatial.Delaunay(points)
        adj = sp.sparse.lil_array((n, n), dtype=np.int32)

        for t in tri.simplices:
            adj[t[0], t[1]] = 1
            adj[t[1], t[2]] = 1
            adj[t[2], t[0]] = 1
            adj[t[1], t[0]] = 1
            adj[t[2], t[1]] = 1
            adj[t[0], t[2]] = 1

        G = nx.from_scipy_sparse_array(adj)
        graphs.append(G)

    return graphs


def generate_tree_graphs(num_graphs, min_size, max_size=None, seed=0):
    """Generate tree graphs using the networkx library."""
    rng = np.random.default_rng(seed)
    graphs: list[nx.Graph] = []

    max_size = max_size or min_size

    for _ in range(num_graphs):
        n = rng.integers(min_size, max_size, endpoint=True)
        G = nx.random_tree(n, seed=rng)
        graphs.append(G)

    return graphs


def generate_sbm_graphs(
        num_graphs,
        min_size,
        max_size=None,
        min_community_size=20,
        max_community_size=40,
        seed=0):
    """Generate SBM graphs using the networkx library."""
    rng = np.random.default_rng(seed)
    graphs: list[nx.Graph] = []

    max_size = max_size or min_size

    while len(graphs) < num_graphs:
        n = rng.integers(min_size, max_size, endpoint=True)
        community_sizes = rng.integers(
            min_community_size, max_community_size, size=ceil(n / min_community_size))
        community_sizes = community_sizes[
            :np.searchsorted(community_sizes.cumsum(), n - max_community_size, side="right") + 2]
        community_sizes[-1] = n - community_sizes[:-1].sum()
        diff = min_community_size - community_sizes[-1]

        while diff > 0:
            diff_inds, *_ = (np.nonzero(community_sizes[:-1] > min_community_size))[:diff]
            community_sizes[diff_inds] -= 1
            community_sizes[-1] += len(diff_inds)
            diff -= len(diff_inds)

        num_communities = len(community_sizes)

        probs = np.ones([num_communities, num_communities]) * 0.005
        probs[np.arange(num_communities), np.arange(num_communities)] = 0.3
        g = nx.stochastic_block_model(community_sizes, probs, seed=rng)

        if nx.is_connected(g):
            graphs.append(g)

    return graphs


def generate_lobster_graphs(
        num_graphs,
        min_size,
        max_size=None,
        p1=0.7,
        p2=0.7,
        seed=0):
    rng = np.random.default_rng(seed)
    graphs: list[nx.Graph] = []

    max_size = max_size or min_size

    base_prob = 1 / (1  + p1 / ((1 - p1) * (1 - p2)))
    min_n = round(min_size * base_prob)
    max_n = round(max_size * base_prob)

    while len(graphs) < num_graphs:
        n = rng.integers(min_n, max_n, endpoint=True)
        g = nx.random_lobster(n, p1, p2, rng)

        if min_size <= g.number_of_nodes() <= max_size:
            graphs.append(g)

    return graphs


def generate_erdos_renyi_graphs(
        num_graphs,
        min_size,
        max_size=None,
        p=None,
        seed=0):
    p = p or 2 / (min_size - 1)
    rng = np.random.default_rng(seed)
    ns = rng.integers(min_size, max_size, num_graphs, endpoint=True)
    graphs: list[nx.Graph] = [
        nx.erdos_renyi_graph(n, p, rng) for n in ns]

    return graphs
