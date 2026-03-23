import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph
from sklearn.model_selection import train_test_split
import torch_geometric

import utils as utils
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


rna_vocab = {
    'A': 0,
    'C': 1,
    'D': 2,
    'G': 3,
    'K': 4,
    'M': 5,
    'N': 6,
    'R': 7,
    'S': 8,
    'U': 9,
    'V': 10,
    'W': 11,
    'Y': 12,
}


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def encode_rna(seq: str, vocab: dict, unknown: int = -1) -> np.ndarray:
    """
    Convert an RNA string into a 1-D numpy array of integer IDs.

    Parameters
    ----------
    seq : str
        RNA sequence (e.g. "AGCNY").
    vocab : dict, optional
        Mapping from character → int. Defaults to `rna_vocab`.
    unknown : int, optional
        ID to use for any char not found in `vocab`. Defaults to -1.

    Returns
    -------
    np.ndarray
        Array of dtype int with one element per character.
    """
    return np.array([vocab.get(ch, unknown) for ch in seq], dtype=int)


class BPRNADataset(InMemoryDataset):
    # raw_url = (
    #     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
    #     "molnet_publish/qm9.zip"
    # )
    # raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    # processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    def __init__(
        self,
        dataset_name,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.num_graphs = len(self.data.n_nodes)

    @property
    def split_file_name(self):
        return ["bpRNA_structure_TR0.npz", "bpRNA_structure_TS0.npz", "bpRNA_structure_VL0.npz", ]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def raw_file_names(self):
        return ["bpRNA_structure.npz", "bpRNA.csv"]

    @property
    def processed_file_names(self):
        return [self.split + ".pt"]

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        # try:
        #     import rdkit  # noqa

        #     file_path = download_url(self.raw_url, self.raw_dir)
        #     extract_zip(file_path, self.raw_dir)
        #     os.unlink(file_path)

        #     file_path = download_url(self.raw_url2, self.raw_dir)
        #     os.rename(
        #         osp.join(self.raw_dir, "3195404"),
        #         osp.join(self.raw_dir, "uncharacterized.txt"),
        #     )
        # except ImportError:
        #     path = download_url(self.processed_url, self.raw_dir)
        #     extract_zip(path, self.raw_dir)
        #     os.unlink(path)

        # if files_exist(self.split_paths):
        #     return

        # # dataset read
        # dataset = pd.read_csv(self.raw_paths[1])

        # # ── 2. stratified 80 / 10 / 10 split ---------------------------------------
        # train, temp = train_test_split(
        #     dataset,
        #     test_size=0.2,                  # leave 20 % for val+test
        #     stratify=dataset["data_name"],  # balance the three RNA types
        #     random_state=42,
        # )

        # val, test = train_test_split(
        #     temp,
        #     test_size=0.5,                  # half of the 20 % → 10 % each
        #     stratify=temp["data_name"],
        #     random_state=42,
        # )

        # # (Optional) shuffle each frame once more for good measure
        # train = train.sample(frac=1, random_state=42).reset_index(drop=True)
        # val   = val.sample(frac=1, random_state=42).reset_index(drop=True)
        # test  = test.sample(frac=1, random_state=42).reset_index(drop=True)

        # # ── 3. save ----------------------------------------------------------------
        # train.to_csv(os.path.join(self.raw_dir, "train.csv"), index=False)
        # val.to_csv  (os.path.join(self.raw_dir, "val.csv"),   index=False)
        # test.to_csv (os.path.join(self.raw_dir, "test.csv"),  index=False)

        # print(
        #     "Done!\n"
        #     "train:", dict(train["data_name"].value_counts()),
        #     "\nval  :", dict(val["data_name"].value_counts()),
        #     "\ntest :", dict(test["data_name"].value_counts()),
        # )
        pass


    def process(self):
        file_idx = {"train": 0, "val": 1, "test": 2}
        rna_structure = np.load(self.split_paths[file_idx[self.split]], allow_pickle=True)  # all the rna structures
        raw_dataset = pd.read_csv(self.raw_paths[1], index_col=0)  # all the rna sequences

        data_list = []
        # import pdb; pdb.set_trace()
        # set 'file_name' as index
        raw_dataset.set_index("file_name", inplace=True)

        for i, rna in enumerate(tqdm(dict(rna_structure).keys())):
            rna_seq = raw_dataset.loc[rna, "seq"]
            rna_adj = torch.Tensor(rna_structure[rna])  # adjacency matrix

            # Convert sequence to integer encoding
            x = encode_rna(rna_seq, rna_vocab)
            x = F.one_hot(torch.tensor(x), num_classes=len(rna_vocab)).float()
            y = torch.zeros([1, 0]).float()
            # import pdb; pdb.set_trace()

            n = rna_adj.shape[-1]
            edge_index, _ = torch_geometric.utils.dense_to_sparse(rna_adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            # if i > 10:  # debug mode
            #     break

        torch.save(self.collate(data_list), self.processed_paths[0])


class BPRNADataModule(AbstractDataModule):
    '''
    TODO
    This class needs to be adapted with domain specific prior information. @Vincent
    '''
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {
            "train": BPRNADataset(
                dataset_name=self.cfg.dataset.name, split="train", root=root_path
            ),
            "val": BPRNADataset(
                dataset_name=self.cfg.dataset.name, split="val", root=root_path
            ),
            "test": BPRNADataset(
                dataset_name=self.cfg.dataset.name, split="test", root=root_path
            ),
        }
        # import pdb; pdb.set_trace()

        train_len = len(datasets["train"].data.n_nodes)
        val_len = len(datasets["val"].data.n_nodes)
        test_len = len(datasets["test"].data.n_nodes)
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class BPRNADatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.dataset_name = datamodule.inner.dataset_name
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
