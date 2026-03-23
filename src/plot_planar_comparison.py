import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def main() -> None:
    prefix = "/home/ubuntu/repos/defog/src/adjs"
    fontsize = "xx-large"
    fontweight = "bold"
    node_color = "blue"

    data = [
        ("GraphLE", np.loadtxt(f"{prefix}/graphle_planar-y_64.txt")),
        ("GraphLE", np.loadtxt(f"{prefix}/graphle_planar-y_128.txt")),
        ("GraphLE", np.loadtxt(f"{prefix}/graphle_planar-n_256.txt")),
        ("DeFoG", np.loadtxt(f"{prefix}/defog_planar-y_64.txt")),
        ("DeFoG", np.loadtxt(f"{prefix}/defog_planar-n_128.txt")),
        ("DeFoG", np.loadtxt(f"{prefix}/defog_planar-n_256.txt")),
        ("GruM", np.loadtxt(f"{prefix}/grum_planar-y_64.txt")),
        ("GruM", np.loadtxt(f"{prefix}/grum_planar-n_128.txt")),
        ("GruM", np.loadtxt(f"{prefix}/grum_planar-n_256.txt")),
    ]
    graph_data = [(name, nx.from_numpy_array(arr)) for name, arr in data]

    grid_map = {}
    unique_labels = []
    unique_sizes = set()

    for label, g in graph_data:
        size = g.number_of_nodes()
        if label not in unique_labels:
            unique_labels.append(label)
        unique_sizes.add(size)
        grid_map[(label, size)] = g

    sorted_sizes = sorted(list(unique_sizes))
    n_rows = len(unique_labels)
    n_cols = len(sorted_sizes)
    base_size = 4
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(base_size * 5, base_size * 3),
        squeeze=False,
        gridspec_kw={'wspace': 0., 'hspace': 0.}
    )

    for r, label in enumerate(unique_labels):
        for c, size in enumerate(sorted_sizes):
            ax = axes[r][c]
            graph = grid_map[(label, size)]
            is_planar, certificate = nx.check_planarity(graph, counterexample=True)

            if is_planar:
                pos = nx.planar_layout(graph)
                edge_color = "grey"
                width = 1.
                text = "checkmark"
                text_color = "green"
            else:
                pos = nx.spring_layout(graph, seed=0)
                subgraph_edges = set(certificate.edges())
                edge_color = ["red" if edge in subgraph_edges else "grey" for edge in graph.edges()]
                width = [2. if edge in subgraph_edges else 1. for edge in graph.edges()]
                text = "times"
                text_color = "red"
            nx.draw(
                graph, pos, ax=ax,
                node_size=15, edge_color=edge_color, node_color=node_color, width=width)
            ax.text(
                0.99, 0.01, f"$\\{text}$",
                horizontalalignment="right",
                verticalalignment="bottom",
                fontsize=24,
                fontweight=fontweight,
                transform=ax.transAxes,
                color=text_color,
            )

            ax.axis("on")
            ax.spines['top'].set_visible(r > 0)
            ax.spines['bottom'].set_visible(r < n_rows - 1)
            ax.spines['left'].set_visible(c > 0)
            ax.spines['right'].set_visible(c < n_cols - 1)

            if r == 0:
                ax.set_title(
                    f"N={size}",
                    fontsize=fontsize,
                    fontweight=fontweight,
                )

            if c == 0:
                ax.set_ylabel(
                    label,
                    fontsize=fontsize,
                    fontweight=fontweight,
                )

    fig.savefig(f"planar_comparison.pdf", pad_inches=0., bbox_inches="tight")


if __name__ == "__main__":
    main()
