import pickle
from pathlib import Path

import networkx as nx

from src.datasets.graph_generators import generate_erdos_renyi_graphs

N = 64
TRAIN_LEN = 128
VAL_LEN = 32
TEST_LEN = 40
DATA_PATH = Path(__file__).parent


def main() -> None:
    graphs = generate_erdos_renyi_graphs(TRAIN_LEN + VAL_LEN + TEST_LEN, N)
    ds = [nx.density(g) for g in graphs]
    print(P, min(ds), max(ds))
    dataset = {
        "train": graphs[:TRAIN_LEN],
        "val": graphs[TRAIN_LEN:-TEST_LEN],
        "test": graphs[-TEST_LEN:],
    }

    with open(DATA_PATH / "erdosrenyi.pkl", "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
