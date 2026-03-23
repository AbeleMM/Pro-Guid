import graph_tool.all as gt

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice
from typing import Optional, Sequence, cast
import time
import wandb
import os

import networkx as nx
import numpy as np
import pickle
import random
# import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from torch.distributions.categorical import Categorical

from models.transformer_model import GraphTransformer

from metrics.train_metrics import TrainLossDiscrete
from src import utils
from analysis.spectre_utils import is_lobster_graph, is_sbm_graph
from flow_matching.noise_distribution import NoiseDistribution
from flow_matching.time_distorter import TimeDistorter
from flow_matching.rate_matrix import RateMatrixDesigner
from flow_matching.utils import p_xt_g_x1
from flow_matching import flow_matching_utils


class GraphDiscreteFlowModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
        test_labels=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.name = f"{cfg.dataset.name}_{cfg.general.name}"
        # self.model_dtype = torch.float32
        self.conditional = cfg.general.conditional
        self.control = cfg.control
        self.test_labels = test_labels

        # number of steps used for sampling
        self.sample_T = cfg.sample.sample_steps

        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.node_dist = dataset_infos.nodes_dist
        print("max num nodes: ", len(self.node_dist.prob) - 1)
        print("min num nodes: ", torch.where(self.node_dist.prob > 0)[0][0].item())

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        if self.cfg.sample.sampler == "e2e-grad":
            from src.datasets.graph_generators import (generate_lobster_graphs,
                                                       generate_planar_graphs,
                                                       generate_sbm_graphs,
                                                       generate_tree_graphs)

            if self.cfg.general.name == "planar":
                gen_func = generate_planar_graphs
            elif self.cfg.general.name == "tree":
                gen_func = generate_tree_graphs
            elif self.cfg.general.name == "sbm":
                gen_func = generate_sbm_graphs
            elif self.cfg.general.name == "lobster":
                gen_func = generate_lobster_graphs
            else:
                raise NotImplementedError

            densities_mean = np.array([
                nx.density(g)
                for g in gen_func(100, self.cfg.sample.num_nodes)
            ]).mean()
            dataset_infos.edge_types[0] = 1 - densities_mean
            dataset_infos.edge_types[1] = densities_mean

        self.noise_dist = NoiseDistribution(cfg.model.transition, dataset_infos)
        self.limit_dist = self.noise_dist.get_limit_dist()

        # add virtual class when absorbing state refers to a new class
        self.noise_dist.update_input_output_dims(self.input_dims)
        self.noise_dist.update_dataset_infos(self.dataset_info)

        self.train_loss = TrainLossDiscrete(
            self.cfg.model.lambda_train,
        )

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.save_hyperparameters(
            ignore=[
                "train_metrics",
                "sampling_metrics",
            ],
        )

        # logging
        self.start_epoch_time = 0.
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.val_counter = 0
        self.adapt_counter = 0

        # time distortor for both training and sampling steps
        self.time_distorter = TimeDistorter(
            train_distortion=cfg.train.time_distortion,
            sample_distortion=cfg.sample.time_distortion,
            alpha=1,
            beta=1,
        )

        # rate matrix designer
        self.rate_matrix_designer = RateMatrixDesigner(
            rdb=self.cfg.sample.rdb,
            rdb_crit=self.cfg.sample.rdb_crit,
            eta=self.cfg.sample.eta,
            omega=self.cfg.sample.omega,
            limit_dist=self.limit_dist,
        )

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        if self.conditional:
            if torch.rand(1) < 0.1:
                data.y = torch.ones_like(data.y, device=self.device) * -1

        dense_data, node_mask = utils.to_dense(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self(noisy_data, extra_data, node_mask)

        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            log=i % self.log_every_steps == 0,
        )

        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=i % self.log_every_steps == 0,
        )

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print(
            "Size of the input features",
            self.input_dims["X"],
            self.input_dims["E"],
            self.input_dims["y"],
        )
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
            f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
            f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
            f" -- {time.time() - self.start_epoch_time:.1f}s "
        )
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}"
        )
        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)

    def on_validation_epoch_start(self) -> None:
        print("Starting validation...")
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        return

    def on_validation_epoch_end(self) -> None:
        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            print("Starting to sample")
            samples, labels = self.sample(
                is_test=False, save_samples=False, save_visualization=True
            )
            to_log = self.evaluate_samples(
                samples=samples, labels=labels, is_test=False
            )

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"val_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

        self.print("Finished validation.")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.sampling_metrics.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        return

    def on_test_epoch_end(self) -> None:

        if self.cfg.sample.search:
            print("Starting sampling optimization...")
            self.search_hyperparameters()
        else:
            print("Starting to sample")
            samples, labels = self.sample(
                is_test=True,
                save_samples=self.cfg.general.save_samples,
                save_visualization=True,
            )
            to_log = self.evaluate_samples(samples=samples, labels=labels, is_test=True)

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"test_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

            self.print("Finished testing.")

    def sample(self, is_test, save_samples, save_visualization):

        # Load generated samples if they exist
        if self.cfg.general.generated_path:
            self.print("Loading generated samples...")
            with open(self.cfg.general.generated_path, "rb") as f:
                samples = pickle.load(f)
            # Set labels to None
            labels = [None] * len(samples)
            return samples, labels

        # Otherwise, generate new samples
        # self.cfg.general.num_sample_fold = 4
        if is_test:
            samples_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save

        else:
            samples_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

        samples = []
        labels = []
        graph_id = 0
        bs = 2 * self.cfg.train.batch_size

        self.limit_dist.to_device(self.device)

        if self.cfg.sample.sampler == "1-node":
            batch_sampler = self.sample_batch_merge_one_node
        elif self.cfg.sample.sampler == "1-edge":
            batch_sampler = self.sample_batch_add_one_edge
        elif self.cfg.sample.sampler == "manifold":
            batch_sampler = self.sample_batch_manifold
        elif self.cfg.sample.sampler == "manifold-ego":
            batch_sampler = self.sample_batch_manifold_ego
        elif self.cfg.sample.sampler == "quad":
            batch_sampler = self.sample_batch_quad
        elif self.cfg.sample.sampler == "expansion":
            batch_sampler = self.sample_batch_expansion
        elif self.cfg.sample.sampler == "expansion-contract":
            batch_sampler = self.sample_batch_expansion_local
        elif self.cfg.sample.sampler == "cond-sep":
            batch_sampler = self.sample_batch_cond_sep
        elif self.cfg.sample.sampler == "cycles":
            batch_sampler = self.sample_batch_cycles
        elif self.cfg.sample.sampler == "grad-guid":
            batch_sampler = self.sample_batch_e2e_grad
        elif self.cfg.sample.sampler == "pro-guid":
            batch_sampler = self.sample_batch_rule_cond
        else:
            batch_sampler = self.sample_batch

        while samples_left_to_generate > 0:
            self.print(
                f"Samples left to generate: {samples_left_to_generate}/"
                f"{samples_to_generate}",
                # end="",
                flush=True,
            )
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            num_chain_steps = min(self.number_chain_steps, self.sample_T)
            cur_samples, cur_labels = batch_sampler(
                graph_id,
                to_generate,
                num_nodes=self.cfg.sample.num_nodes,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=num_chain_steps,
                save_visualization=save_visualization,
            )
            samples.extend(cur_samples)
            labels.extend(cur_labels)

            graph_id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        if save_samples:
            self.print("Saving the generated graphs")

            # saving in txt version
            filename = "graphs.txt"
            with open(filename, "w") as f:
                for item in samples:
                    f.write(f"N={item[0].shape[0]}\n")
                    atoms = item[0].tolist()
                    f.write("X: \n")
                    for at in atoms:
                        f.write(f"{at} ")
                    f.write("\n")
                    f.write("E: \n")
                    for bond_list in item[1]:
                        for bond in bond_list:
                            f.write(f"{bond} ")
                        f.write("\n")
                    f.write("\n")

            # saving in pkl version
            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(samples, f)

            print("Generated graphs saved.")

        return samples, labels

    def evaluate_samples(
        self,
        samples,
        labels,
        is_test,
        save_filename="",
    ):
        print("Computing sampling metrics...")

        to_log = {}
        samples_to_evaluate = self.cfg.general.final_model_samples_to_generate
        if is_test:
            for i in range(self.cfg.general.num_sample_fold):
                cur_samples = samples[
                    i * samples_to_evaluate : (i + 1) * samples_to_evaluate
                ]
                cur_labels = labels[
                    i * samples_to_evaluate : (i + 1) * samples_to_evaluate
                ]

                cur_to_log = self.sampling_metrics(
                    cur_samples,
                    ref_metrics=self.dataset_info.ref_metrics,
                    name=f"{self.name}_{i}",
                    current_epoch=self.current_epoch,
                    val_counter=-1,
                    test=is_test,
                    local_rank=self.local_rank,
                    labels=cur_labels if self.conditional else None,
                    n_nodes=self.cfg.sample.num_nodes,
                )

                if i == 0:
                    to_log = {i: [cur_to_log[i]] for i in cur_to_log}
                else:
                    to_log = {i: to_log[i] + [cur_to_log[i]] for i in cur_to_log}

                filename = os.path.join(
                    os.getcwd(),
                    f"epoch{self.current_epoch}_res_fold{i}_{save_filename}.txt",
                )
                with open(filename, "w") as file:
                    for key, value in cur_to_log.items():
                        file.write(f"{key}: {value}\n")

            to_log = {
                i: (np.array(to_log[i]).mean(), np.array(to_log[i]).std())
                for i in to_log
            }
        else:
            to_log = self.sampling_metrics(
                samples,
                ref_metrics=self.dataset_info.ref_metrics,
                name=self.cfg.general.name,
                current_epoch=self.current_epoch,
                val_counter=-1,
                test=is_test,
                local_rank=self.local_rank,
                labels=labels if self.conditional else None,
            )

        return to_log

    def apply_noise(self, X, E, y, node_mask, t=None):
        """Sample noise and apply it to the data."""

        # Sample a timestep t.
        bs = X.size(0)
        if t is None:
            t_float = self.time_distorter.train_ft(bs, self.device)
        else:
            t_float = t

        # sample random step
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        prob_X_t, prob_E_t = p_xt_g_x1(
            X1=X_1_label, E1=E_1_label, t=t_float, limit_dist=self.limit_dist
        )

        # step 4 - sample noised data
        sampled_t = flow_matching_utils.sample_discrete_features(
            probX=prob_X_t, probE=prob_E_t, node_mask=node_mask
        )
        noise_dims = self.noise_dist.get_noise_dims()
        X_t = F.one_hot(sampled_t.X, num_classes=noise_dims["X"])
        E_t = F.one_hot(sampled_t.E, num_classes=noise_dims["E"])

        # step 5 - create the PlaceHolder
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t": t_float,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        # import pdb; pdb.set_trace()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
        save_visualization: bool = True,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        if self.conditional:
            if "qm9" in self.cfg.dataset.name:
                y = self.test_labels
                perm = torch.randperm(y.size(0))
                idx = perm[:100]
                condition = y[idx]
                condition = condition.to(self.device)
                z_T.y = condition.repeat([10, 1])[:batch_size, :]
            elif "tls" in self.cfg.dataset.name:
                z_T.y = torch.zeros(batch_size, 1, device=self.device)
                z_T.y[: batch_size // 2] = 1
            else:
                raise NotImplementedError
        X, E, y = z_T.X, z_T.E, z_T.y

        # Init chain storing variables
        assert (E == torch.transpose(E, 1, 2)).all()
        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps + 1, keep_chain, E.size(1), E.size(2))
        )
        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)
        # chain_E_prob = torch.zeros(chain_E_size)
        chain_times = torch.zeros((number_chain_steps + 1, keep_chain))
        chain_time_unit = 1 / number_chain_steps

        # Store initial graph
        if keep_chain > 0:
            sampled_initial = z_T.mask(node_mask, collapse=True)
            chain_X[0] = sampled_initial.X[:keep_chain]
            chain_E[0] = sampled_initial.E[:keep_chain]
            chain_times[0] = torch.zeros((keep_chain))

        for t_int in trange(0, self.cfg.sample.sample_steps):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / self.cfg.sample.sample_steps
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / self.cfg.sample.sample_steps

            # using round for precision
            write_index = int(np.ceil(np.round(s_norm[0].item() / chain_time_unit, 6)))

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s, E_prob = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
            # chain_E_prob[write_index] = E_prob[:keep_chain]
            chain_times[write_index] = s_norm.flatten()[:keep_chain]

            if self.control.condition and t_int == self.cfg.sample.sample_steps - 1:
                X, A_E = drifted_project(
                    X.float(), E[:, :, :, 1].float(),
                    t_int + 1, self.cfg.sample.sample_steps,
                    self.control.condition, self.control.sched_gamma, self.control.sched_params)

                if self.control.resample:
                    A_E_upr = torch.triu(A_E).bernoulli()
                    A_E = A_E_upr + A_E_upr.transpose(1, 2)
                else:
                    A_E = (A_E > 0.5).type_as(A_E)

                E[:, :, :, 1] = A_E
                E[:, :, :, 0] = 1 - A_E

        # Sample
        X, E, y = discrete_sampled_s.X, discrete_sampled_s.E, discrete_sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            # Repeat last frame 10x to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            # chain_E_prob = torch.cat(
            #     [chain_E_prob, chain_E_prob[-1:].repeat(10, 1, 1, 1)], dim=0)
            chain_times = torch.cat(
                [chain_times, chain_times[-1:].repeat(10, 1)], dim=0
            )
            assert chain_X.size(0) == (number_chain_steps + 1 + 10)

        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)
        chain_X, chain_E, _ = self.noise_dist.ignore_virtual_classes(
            chain_X, chain_E, y
        )

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            # Visualize chains
            self.print("Visualizing chains...")
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(
                    current_path,
                    f"chains/{self.cfg.general.name}/"
                    f"epoch{self.current_epoch}/"
                    f"chains/molecule_{batch_id + i}",
                )
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    chain_times_arr = chain_times[:, i].numpy()
                    _ = self.visualization_tools.visualize_chain(
                        result_path,
                        chain_X[:, i, :].numpy(),
                        chain_E[:, i, :].numpy(),
                        chain_times_arr,
                    )
                    # self.visualization_tools.visualize_chain_edge_prob(
                    #     result_path,
                    #     chain_E_prob[:, i, :].numpy(),
                    #     chain_times_arr,
                    # )
                self.print(
                    "\r{}/{} complete".format(i + 1, num_molecules), end="", flush=True
                )
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list

    @torch.no_grad()
    def sample_batch_merge_one_node(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        gen_size = cast(int, self.node_dist.prob.numel() - 1)

        if num_nodes is None:
            num_nodes = gen_size

        n_nodes = num_nodes * torch.ones(
            batch_size, device=self.device, dtype=torch.int
        )
        num_elements = round(num_nodes / gen_size)

        if num_elements == 0:
            num_elements = 1

        base_value = num_nodes // num_elements
        remainder = num_nodes % num_elements

        g_sizes = [base_value + 1] * remainder + [base_value] * (num_elements - remainder)

        for i in range(1, len(g_sizes)):
            g_sizes[i] += 1

        size_acc = 0
        X_fin = torch.zeros((batch_size, num_nodes), dtype=torch.int, device="cpu")
        E_fin = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.int, device=self.device)

        for size in g_sizes:
            node_mask = torch.ones(
                (batch_size, size), device=self.device, dtype=torch.bool)

            # Sample noise -- z has size (n_samples, n_nodes, n_features)
            z_T = flow_matching_utils.sample_discrete_feature_noise(
                limit_dist=self.limit_dist, node_mask=node_mask
            )
            X, E, y = z_T.X, z_T.E, z_T.y

            assert (E == torch.transpose(E, 1, 2)).all()

            for t_int in trange(0, self.cfg.sample.sample_steps):
                # this state
                t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
                t_norm = t_array / self.cfg.sample.sample_steps
                if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                    # to avoid failure mode of absorbing transition, add epsilon
                    t_norm = t_norm + 1e-6
                # next state
                s_array = t_array + 1
                s_norm = s_array / self.cfg.sample.sample_steps

                # Distort time
                t_norm = self.time_distorter.sample_ft(
                    t_norm, self.cfg.sample.time_distortion
                )
                s_norm = self.time_distorter.sample_ft(
                    s_norm, self.cfg.sample.time_distortion
                )

                # Sample z_s
                sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                    t_norm,
                    s_norm,
                    X,
                    E,
                    y,
                    node_mask,
                )

                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            start = size_acc - (size_acc > 0)
            stop = start + size
            E_fin[:, start : stop, start : stop] = discrete_sampled_s.E
            size_acc += size - (size_acc > 0)

        X, E, y = X_fin, E_fin, discrete_sampled_s.y
        print(X.shape, E.shape, y.shape)
        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list


    @torch.no_grad()
    def sample_batch_add_one_edge(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        gen_size = cast(int, self.node_dist.prob.numel() - 1)

        if num_nodes is None:
            num_nodes = gen_size

        n_nodes = num_nodes * torch.ones(
            batch_size, device=self.device, dtype=torch.int
        )
        num_elements = round(num_nodes / gen_size)

        if num_elements == 0:
            num_elements = 1

        base_value = num_nodes // num_elements
        remainder = num_nodes % num_elements

        g_sizes = [base_value + 1] * remainder + [base_value] * (num_elements - remainder)

        size_acc = 0
        X_fin = torch.zeros((batch_size, num_nodes), dtype=torch.int, device="cpu")
        E_fin = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.int, device=self.device)

        for size in g_sizes:
            node_mask = torch.ones(
                (batch_size, size), device=self.device, dtype=torch.bool)

            # Sample noise -- z has size (n_samples, n_nodes, n_features)
            z_T = flow_matching_utils.sample_discrete_feature_noise(
                limit_dist=self.limit_dist, node_mask=node_mask
            )
            X, E, y = z_T.X, z_T.E, z_T.y

            assert (E == torch.transpose(E, 1, 2)).all()

            for t_int in trange(0, self.cfg.sample.sample_steps):
                # this state
                t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
                t_norm = t_array / self.cfg.sample.sample_steps
                if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                    # to avoid failure mode of absorbing transition, add epsilon
                    t_norm = t_norm + 1e-6
                # next state
                s_array = t_array + 1
                s_norm = s_array / self.cfg.sample.sample_steps

                # Distort time
                t_norm = self.time_distorter.sample_ft(
                    t_norm, self.cfg.sample.time_distortion
                )
                s_norm = self.time_distorter.sample_ft(
                    s_norm, self.cfg.sample.time_distortion
                )

                # Sample z_s
                sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                    t_norm,
                    s_norm,
                    X,
                    E,
                    y,
                    node_mask,
                )

                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            E_fin[:, size_acc : size_acc + size, size_acc : size_acc + size] = discrete_sampled_s.E

            if size_acc > 0:
                E_fin[:, size_acc - 1, size_acc] = E_fin[:, size_acc, size_acc - 1] = 1

            size_acc += size

        X, E, y = X_fin, E_fin, discrete_sampled_s.y
        print(X.shape, E.shape, y.shape)
        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list


    @torch.no_grad()
    def sample_batch_manifold(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        gen_size = cast(int, self.node_dist.prob.numel() - 1)

        if num_nodes is None:
            num_nodes = gen_size

        n_nodes = num_nodes * torch.ones(
            batch_size, device=self.device, dtype=torch.int
        )
        num_elements = round(num_nodes / gen_size)

        num_elements += (num_elements == 0)

        base_value = num_nodes // num_elements
        remainder = num_nodes % num_elements

        g_sizes = [base_value + 1] * remainder + [base_value] * (num_elements - remainder)

        for i in range(1, len(g_sizes)):
            g_sizes[i] += 1
        #

        # len(g_sizes) x batch_size
        E_batches: list[list[nx.Graph]] = [[]] * len(g_sizes)

        for i, size in enumerate(g_sizes):
            node_mask = torch.ones(
                (batch_size, size), device=self.device, dtype=torch.bool)

            # Sample noise -- z has size (n_samples, n_nodes, n_features)
            z_T = flow_matching_utils.sample_discrete_feature_noise(
                limit_dist=self.limit_dist, node_mask=node_mask
            )
            X, E, y = z_T.X, z_T.E, z_T.y

            assert (E == torch.transpose(E, 1, 2)).all()

            for t_int in trange(0, self.cfg.sample.sample_steps):
                # this state
                t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
                t_norm = t_array / self.cfg.sample.sample_steps
                if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                    # to avoid failure mode of absorbing transition, add epsilon
                    t_norm = t_norm + 1e-6
                # next state
                s_array = t_array + 1
                s_norm = s_array / self.cfg.sample.sample_steps

                # Distort time
                t_norm = self.time_distorter.sample_ft(
                    t_norm, self.cfg.sample.time_distortion
                )
                s_norm = self.time_distorter.sample_ft(
                    s_norm, self.cfg.sample.time_distortion
                )

                # Sample z_s
                sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                    t_norm,
                    s_norm,
                    X,
                    E,
                    y,
                    node_mask,
                )

                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            _, E_disc, _ = self.noise_dist.ignore_virtual_classes(
                discrete_sampled_s.X, discrete_sampled_s.E, discrete_sampled_s.y)
            E_batches[i] = [nx.from_numpy_array(adj.numpy(force=True)) for adj in E_disc]

        full_graphs, *ext_batches = E_batches

        if self.cfg.general.name == "planar":
            val_func = nx.is_planar
        elif self.cfg.general.name == "tree":
            val_func = nx.is_tree
        elif self.cfg.general.name == "sbm":
            val_func = partial(is_sbm_graph, refinement_steps=50)
        elif self.cfg.general.name == "lobster":
            val_func = is_lobster_graph
        else:
            val_func = lambda _: True

        for ext_batch in ext_batches:
            for i, (full_graph, ext_graph) in enumerate(zip(full_graphs, ext_batch)):
                # for a connected graph, is equal to returning a copy
                # for an unconnected graph, it gives a connected copy
                full_g, ext_g = min_connected_graph(full_graph), min_connected_graph(ext_graph)

                node_full_g: int = sorted(full_g.degree, key=lambda x: x[1])[len(full_g) // 2][0]
                node_ext_g: int = sorted(ext_g.degree, key=lambda x: x[1])[len(ext_g) // 2][0]
                label_offset = max(full_g.nodes) + 1

                ext_g = nx.relabel_nodes(
                    ext_g,
                    (
                        dict(zip(ext_g.nodes, range(label_offset, label_offset + len(ext_g)))) |
                        {node_ext_g: node_full_g}
                    )
                )

                full_g_ov: nx.Graph = nx.ego_graph(full_g, node_full_g).copy()
                ext_g_ov: nx.Graph = nx.ego_graph(ext_g, node_full_g).copy()
                full_g_ov.add_nodes_from(ext_g_ov.nodes)
                ext_g_ov.add_nodes_from(full_g_ov.nodes)
                sorted_ov_nodes = sorted(full_g_ov.nodes)

                full_ov_lapl = nx.normalized_laplacian_matrix(full_g_ov, sorted_ov_nodes).toarray()
                ext_ov_lapl = nx.normalized_laplacian_matrix(ext_g_ov, sorted_ov_nodes).toarray()

                _, full_ov_eigvecs = np.linalg.eigh(full_ov_lapl)
                _, ext_ov_eigvecs = np.linalg.eigh(ext_ov_lapl)

                grassmann_lapl = \
                    full_ov_lapl - 0.5 * full_ov_eigvecs @ full_ov_eigvecs.T + \
                    ext_ov_lapl - 0.5 * ext_ov_eigvecs @ ext_ov_eigvecs.T
                pseudo_deg_mat = np.diagflat((
                    np.array([d for _, d in full_g_ov.degree]) +
                    np.array([d for _, d in ext_g_ov.degree])
                ) // 2)

                grassmann_weight_adj = pseudo_deg_mat - grassmann_lapl

                rs, cs = np.triu_indices_from(grassmann_weight_adj, k=1)
                values = grassmann_weight_adj[rs, cs]
                mask = values >= grassmann_weight_adj.mean()
                rc_desc_sort_inds = np.argsort(values[mask])[::-1]
                rs = rs[mask][rc_desc_sort_inds]
                cs = cs[mask][rc_desc_sort_inds]

                full_g = nx.compose(full_g, ext_g)

                valid_start = val_func(full_g)

                for r, c in zip(rs, cs):
                    u, v = sorted_ov_nodes[r], sorted_ov_nodes[c]
                    full_g.add_edge(u, v)

                    if valid_start and not val_func(full_g):
                        full_g.remove_edge(u, v)

                full_graphs[i] = full_g

        X = torch.zeros((batch_size, num_nodes), dtype=torch.int, device="cpu")
        E = torch.from_numpy(np.stack([nx.to_numpy_array(g) for g in full_graphs]))

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list

    @torch.no_grad()
    def sample_batch_manifold_ego(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        gen_size = cast(int, self.node_dist.prob.numel() - 1)

        if num_nodes is None:
            num_nodes = gen_size

        n_nodes = num_nodes * torch.ones(
            batch_size, device=self.device, dtype=torch.int
        )
        num_elements = round(num_nodes / gen_size)

        if num_elements == 0:
            num_elements = 1

        base_value = num_nodes // num_elements
        remainder = num_nodes % num_elements

        g_sizes = [base_value + 1] * remainder + [base_value] * (num_elements - remainder)

        for i in range(1, len(g_sizes)):
            g_sizes[i] += 1

        # len(g_sizes) x batch_size
        E_batches: list[list[nx.Graph]] = [[]] * len(g_sizes)

        for i, size in enumerate(g_sizes):
            node_mask = torch.ones(
                (batch_size, size), device=self.device, dtype=torch.bool)

            # Sample noise -- z has size (n_samples, n_nodes, n_features)
            z_T = flow_matching_utils.sample_discrete_feature_noise(
                limit_dist=self.limit_dist, node_mask=node_mask
            )
            X, E, y = z_T.X, z_T.E, z_T.y

            assert (E == torch.transpose(E, 1, 2)).all()

            for t_int in trange(0, self.cfg.sample.sample_steps):
                # this state
                t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
                t_norm = t_array / self.cfg.sample.sample_steps
                if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                    # to avoid failure mode of absorbing transition, add epsilon
                    t_norm = t_norm + 1e-6
                # next state
                s_array = t_array + 1
                s_norm = s_array / self.cfg.sample.sample_steps

                # Distort time
                t_norm = self.time_distorter.sample_ft(
                    t_norm, self.cfg.sample.time_distortion
                )
                s_norm = self.time_distorter.sample_ft(
                    s_norm, self.cfg.sample.time_distortion
                )

                # Sample z_s
                sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                    t_norm,
                    s_norm,
                    X,
                    E,
                    y,
                    node_mask,
                )

                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            _, E_disc, _ = self.noise_dist.ignore_virtual_classes(
                discrete_sampled_s.X, discrete_sampled_s.E, discrete_sampled_s.y)
            E_batches[i] = [nx.from_numpy_array(adj.numpy(force=True)) for adj in E_disc]

        full_graphs, *ext_batches = E_batches

        if self.cfg.general.name == "planar":
            val_func = nx.is_planar
        elif self.cfg.general.name == "tree":
            val_func = nx.is_tree
        elif self.cfg.general.name == "sbm":
            val_func = is_sbm_graph
        elif self.cfg.general.name == "lobster":
            val_func = is_lobster_graph
        else:
            val_func = lambda _: True

        for ext_batch in ext_batches:
            for i, (full_graph, ext_graph) in enumerate(zip(full_graphs, ext_batch)):
                full_g, ext_g = full_graph, ext_graph
                deg_to_ind_full_graph: dict[int, int] = {
                    d: ind for ind, d in cast(Sequence, full_g.degree)}
                deg_to_ind_ext_graph: dict[int, int] = {
                    d: ind for ind, d in cast(Sequence, ext_g.degree)}

                max_common_deg: int = max(
                    set(deg_to_ind_full_graph.keys()) &
                    set(deg_to_ind_ext_graph.keys()))

                ind_full_graph = deg_to_ind_full_graph[max_common_deg]
                ind_ext_graph = deg_to_ind_ext_graph[max_common_deg]

                ego_full = nx.ego_graph(full_g, ind_full_graph)
                ego_ext = nx.ego_graph(ext_g, ind_ext_graph)

                ego_ext_nodes = list(ego_ext.nodes)
                ind = ego_ext_nodes.index(ind_ext_graph)
                ego_ext_nodes[0], ego_ext_nodes[ind] = ego_ext_nodes[ind], ego_ext_nodes[0]
                ego_full_nodes = list(ego_full.nodes)
                ind = ego_full_nodes.index(ind_full_graph)
                ego_full_nodes[0], ego_full_nodes[ind] = ego_full_nodes[ind], ego_full_nodes[0]

                max_label_full_g: int = max(full_g.nodes)
                node_map = (
                    {u: u + max_label_full_g for u in ext_g.nodes} |
                    dict(zip(ego_ext_nodes, ego_full_nodes)))
                ext_g = nx.relabel_nodes(ext_g, node_map)
                ego_ext = nx.relabel_nodes(ego_ext, node_map)

                full_g = nx.compose(full_g, ext_g)
                #
                full_g.remove_edges_from(ego_full.edges)
                full_g.remove_edges_from(ego_ext.edges)
                #
                # full_g.remove_nodes_from(ego_full.nodes)

                ego_full_laplacian = nx.normalized_laplacian_matrix(ego_full).toarray()
                ego_ext_laplacian = nx.normalized_laplacian_matrix(ego_ext).toarray()

                ego_full_eigvecs = np.linalg.eigh(ego_full_laplacian)[1][:, :max_common_deg // 2]
                ego_ext_eigvecs = np.linalg.eigh(ego_ext_laplacian)[1][:, :max_common_deg // 2]

                ego_grassmann_laplacian = \
                    ego_full_laplacian - 0.5 * ego_full_eigvecs @ ego_full_eigvecs.T + \
                    ego_ext_laplacian - 0.5 * ego_ext_eigvecs @ ego_ext_eigvecs.T
                pseudo_deg_mat = np.diagflat((
                    np.fromiter(
                        (w for _, w in cast(Sequence, ego_full.degree)), int, count=max_common_deg + 1) +
                    np.fromiter(
                        (w for _, w in cast(Sequence, ego_ext.degree)), int, count=max_common_deg + 1)
                ) // 2)
                ego_grassmann_weight_adj = pseudo_deg_mat - ego_grassmann_laplacian
                rs, cs = np.nonzero(ego_grassmann_weight_adj)
                sort_indices = np.argsort(-ego_grassmann_weight_adj[rs, cs])
                rs = rs[sort_indices]
                cs = cs[sort_indices]
                node_map = list(ego_full.nodes)

                for r, c in random.sample(list(zip(rs, cs)), len(rs)):
                    u, v = node_map[r], node_map[c]
                    full_g.add_edge(u, v)

                    if not val_func(full_g):
                        full_g.remove_edge(u, v)

                full_g.remove_edges_from(nx.selfloop_edges(full_g))
                full_graphs[i] = full_g

        X = torch.zeros((batch_size, num_nodes), dtype=torch.int, device="cpu")

        for full_g in full_graphs:
            max_label_full_g: int = max(full_g.nodes)
            full_g.add_nodes_from(
                [max_label_full_g + i + 1 for i in range(num_nodes - full_g.number_of_nodes())])

        E = torch.from_numpy(np.stack([nx.to_numpy_array(g) for g in full_graphs]))

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list

    @torch.no_grad()
    def sample_batch_quad(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        gen_size = cast(int, self.node_dist.prob.numel() - 1)

        if num_nodes is None:
            num_nodes = gen_size

        n_nodes = num_nodes * torch.ones(
            batch_size, device=self.device, dtype=torch.int
        )
        g_sizes = [gen_size] * (num_nodes // gen_size + ((num_nodes % gen_size) > 0))**2
        # len(g_sizes) x batch_size
        E_batches: list[list[nx.Graph]] = [[]] * len(g_sizes)

        for i, size in enumerate(g_sizes):
            node_mask = torch.ones(
                (batch_size, size), device=self.device, dtype=torch.bool)

            # Sample noise -- z has size (n_samples, n_nodes, n_features)
            z_T = flow_matching_utils.sample_discrete_feature_noise(
                limit_dist=self.limit_dist, node_mask=node_mask
            )
            X, E, y = z_T.X, z_T.E, z_T.y

            assert (E == torch.transpose(E, 1, 2)).all()

            for t_int in trange(0, self.cfg.sample.sample_steps):
                # this state
                t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
                t_norm = t_array / self.cfg.sample.sample_steps
                if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                    # to avoid failure mode of absorbing transition, add epsilon
                    t_norm = t_norm + 1e-6
                # next state
                s_array = t_array + 1
                s_norm = s_array / self.cfg.sample.sample_steps

                # Distort time
                t_norm = self.time_distorter.sample_ft(
                    t_norm, self.cfg.sample.time_distortion
                )
                s_norm = self.time_distorter.sample_ft(
                    s_norm, self.cfg.sample.time_distortion
                )

                # Sample z_s
                sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                    t_norm,
                    s_norm,
                    X,
                    E,
                    y,
                    node_mask,
                )

                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            _, E_disc, _ = self.noise_dist.ignore_virtual_classes(
                discrete_sampled_s.X, discrete_sampled_s.E, discrete_sampled_s.y)
            E_batches[i] = [nx.from_numpy_array(adj.numpy(force=True)) for adj in E_disc]

        if self.cfg.general.name == "planar":
            val_func = nx.is_planar
        elif self.cfg.general.name == "tree":
            val_func = nx.is_tree
        elif self.cfg.general.name == "lobster":
            val_func = is_lobster_graph
        else:
            val_func = lambda _: True

        full_graphs: list[nx.Graph] = []

        for i in range(batch_size):
            g = nx.union(
                E_batches[0][i],
                nx.relabel_nodes(
                    E_batches[1][i], {v: v + gen_size for v in E_batches[1][i].nodes})
            )

            for u_old, v_old in E_batches[2][i].edges:
                u_new, v_new = u_old, v_old + gen_size
                g.add_edge(u_new, v_new)

                if not val_func(g):
                    g.remove_edge(u_new, v_new)

            for u_old, v_old in E_batches[3][i].edges:
                u_new, v_new = v_old, u_old + gen_size
                g.add_edge(u_new, v_new)

                if not val_func(g):
                    g.remove_edge(u_new, v_new)

            full_graphs.append(g)

        X = torch.zeros((batch_size, num_nodes), dtype=torch.int, device="cpu")
        E = torch.from_numpy(np.stack([nx.to_numpy_array(g) for g in full_graphs]))

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list


    @torch.no_grad()
    def sample_batch_expansion(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        gen_size = cast(int, self.node_dist.prob.numel() - 1)

        if num_nodes is None:
            num_nodes = gen_size

        n_nodes = num_nodes * torch.ones(
            batch_size, device=self.device, dtype=torch.int
        )
        # offset = self.cfg.sample.sample_steps // 4
        # expand_every = (self.cfg.sample.sample_steps - offset) // max(num_nodes - gen_size, 1)
        expand_every = self.cfg.sample.sample_steps // (num_nodes - 1)

        node_mask = torch.ones(
            (batch_size, num_nodes), device=self.device, dtype=torch.bool)

        # Sample noise -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()

        # node_mask[:, gen_size:] = 0
        node_mask[:, 1:] = 0
        current_size: int

        for t_int in trange(0, self.cfg.sample.sample_steps):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / self.cfg.sample.sample_steps
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / self.cfg.sample.sample_steps

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # current_size = min(num_nodes, gen_size + max(0, t_int - offset) // expand_every)
            current_size = min(num_nodes, 1 + t_int // expand_every)
            node_mask[:, :current_size] = 1

            # Sample z_s
            sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        #

        X, E, y = discrete_sampled_s.X, discrete_sampled_s.E, discrete_sampled_s.y
        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)
        print(X.shape, E.shape, y.shape)

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list


    @torch.no_grad()
    def sample_batch_expansion_local(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        gen_size = cast(int, self.node_dist.prob.numel() - 1)

        if num_nodes is None:
            num_nodes = gen_size

        n_nodes = num_nodes * torch.ones(
            batch_size, device=self.device, dtype=torch.int
        )

        node_mask = torch.ones(
            (batch_size, gen_size), device=self.device, dtype=torch.bool)

        # Sample noise -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()

        for t_int in trange(0, self.cfg.sample.sample_steps):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / self.cfg.sample.sample_steps
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / self.cfg.sample.sample_steps

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        for prev_size in trange(gen_size, num_nodes):
            # selected_indices = torch.randint(0, prev_size, (batch_size,), device=self.device)

            E_argmax = E.argmax(-1)
            deg = E_argmax.sum(dim=-1)
            L = torch.diag_embed(deg) - E_argmax
            _, vecs = torch.linalg.eigh(L.float())
            fiedler_vectors = vecs[:, :, 1]
            selected_indices = torch.argmin(fiedler_vectors, dim=1)

            batch_indices = torch.arange(batch_size, device=self.device)
            new_connections = E_argmax[batch_indices, selected_indices].clone()
            new_connections[batch_indices, selected_indices] = 1
            E_argmax_new = F.pad(E_argmax, (0, 1, 0, 1))
            E_argmax_new[:, prev_size, :prev_size] = new_connections
            E_argmax_new[:, :prev_size, prev_size] = new_connections
            E = F.one_hot(E_argmax_new)
            next_size = prev_size + 1
            X = torch.ones((batch_size, next_size, 1), dtype=torch.int, device=self.device)
            # t_int_start = int((prev_size / next_size) * self.cfg.sample.sample_steps)
            node_mask = torch.ones(
                (batch_size, next_size), device=self.device, dtype=torch.bool)

            for t_int in range(0, self.cfg.sample.sample_steps):
                # this state
                t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
                t_norm = t_array / self.cfg.sample.sample_steps
                if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                    # to avoid failure mode of absorbing transition, add epsilon
                    t_norm = t_norm + 1e-6
                # next state
                s_array = t_array + 1
                s_norm = s_array / self.cfg.sample.sample_steps

                # Distort time
                t_norm = self.time_distorter.sample_ft(
                    t_norm, self.cfg.sample.time_distortion
                )
                s_norm = self.time_distorter.sample_ft(
                    s_norm, self.cfg.sample.time_distortion
                )

                # Sample z_s
                sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                    t_norm,
                    s_norm,
                    X,
                    E,
                    y,
                    node_mask,
                )

                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        X, E, y = discrete_sampled_s.X, discrete_sampled_s.E, discrete_sampled_s.y
        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)
        print(X.shape, E.shape, y.shape)

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list


    @torch.no_grad()
    def sample_batch_cond_sep(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        gen_size = cast(int, self.node_dist.prob.numel() - 1)

        if num_nodes is None:
            num_nodes = gen_size

        n_nodes = num_nodes * torch.ones(
            batch_size, device=self.device, dtype=torch.int
        )

        node_mask = torch.ones(
            (batch_size, gen_size), device=self.device, dtype=torch.bool)

        # Sample noise -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()

        for t_int in trange(0, self.cfg.sample.sample_steps):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / self.cfg.sample.sample_steps
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / self.cfg.sample.sample_steps

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        X_full = torch.zeros((batch_size, num_nodes), dtype=torch.int, device="cpu")
        y_full = discrete_sampled_s.y
        graphs = [
            nx.from_numpy_array(E_i.argmax(-1).numpy(force=True))
            for E_i in E
        ]
        y_exp = torch.zeros((1, 0), dtype=torch.int, device=self.device)

        for graph in tqdm(graphs):
            while graph.number_of_nodes() < num_nodes:
                print(graph.number_of_nodes())
                part_a, sep, part_b = planar_separator_bfs(graph)
                min_part = part_a if len(part_a) < len(part_b) else part_b
                added_size = min(int(gen_size * 1) - len(min_part), num_nodes - graph.number_of_nodes())
                new_chunk_graph = graph.subgraph(min_part)
                expanded_size = len(min_part) + added_size
                E_exp = torch.zeros(
                    (1, expanded_size, expanded_size, 2), dtype=torch.long, device=self.device)
                E_exp[:, :, :, 0] = 1
                E_exp[0, :-added_size, :-added_size] = F.one_hot(torch.from_numpy(
                    nx.to_numpy_array(new_chunk_graph)).long())
                X_exp = torch.ones((1, expanded_size, 1), dtype=torch.float, device=self.device)

                node_mask = torch.ones(
                    (1, expanded_size), device=self.device, dtype=torch.bool)
                noisy_data = self.apply_noise(X_exp, E_exp, y_exp, node_mask)
                X, E, y = noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]

                for t_int in range(0, self.cfg.sample.sample_steps):
                    # this state
                    t_array = t_int * torch.ones((1, 1)).type_as(y)
                    t_norm = t_array / self.cfg.sample.sample_steps
                    if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                        # to avoid failure mode of absorbing transition, add epsilon
                        t_norm = t_norm + 1e-6
                    # next state
                    s_array = t_array + 1
                    s_norm = s_array / self.cfg.sample.sample_steps

                    # Distort time
                    t_norm = self.time_distorter.sample_ft(
                        t_norm, self.cfg.sample.time_distortion
                    )
                    s_norm = self.time_distorter.sample_ft(
                        s_norm, self.cfg.sample.time_distortion
                    )

                    # Sample z_s
                    sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                        t_norm,
                        s_norm,
                        X,
                        E,
                        y,
                        node_mask,
                        E_exp[:, :-added_size, :-added_size]
                    )

                    X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

                edge_inds = E.squeeze().argmax(-1).nonzero()
                edge_inds = edge_inds + graph.number_of_nodes() * (edge_inds >= len(min_part))
                graph.add_edges_from((edge_inds).numpy(force=True))

        E_full = torch.stack([
            torch.from_numpy(nx.to_numpy_array(g))
            for g in graphs
        ]).to(self.device)

        X, E, y = self.noise_dist.ignore_virtual_classes(X_full, E_full, y_full)
        print(X.shape, E.shape, y.shape)

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list


    @torch.no_grad()
    def sample_batch_cycles(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        gen_size = cast(int, self.node_dist.prob.numel() - 1)

        if num_nodes is None:
            num_nodes = gen_size

        node_mask = torch.ones(
            (batch_size, gen_size), device=self.device, dtype=torch.bool)

        # Sample noise -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()

        for t_int in trange(0, self.cfg.sample.sample_steps):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / self.cfg.sample.sample_steps
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / self.cfg.sample.sample_steps

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        X, E, y = discrete_sampled_s.X, discrete_sampled_s.E, discrete_sampled_s.y
        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)

        graphs = [
            nx.from_numpy_array(E_i.numpy(force=True))
            for E_i in E
        ]
        rand = np.random.default_rng(0)

        for graph in tqdm(graphs):
            cycles = nx.cycle_basis(graph)
            cycle_lens, cycle_len_probs = np.unique([len(c) for c in cycles], return_counts=True)
            cycle_len_probs = cycle_len_probs / sum(cycle_len_probs)

            while graph.number_of_nodes() < num_nodes:
                node = rand.choice(np.array(graph.nodes))
                # neigh = list(graph.neighbors(node))
                # cycle_len = rand.choice(cycle_lens, p=cycle_len_probs)
                cycle_len = 3
                next_node = graph.number_of_nodes()
                # graph.remove_node(node)
                new_nodes = np.array([node, *(next_node + i for i in range(cycle_len - 1))])
                graph.add_edges_from(zip(new_nodes, np.roll(new_nodes, -1)))
                # graph.add_edges_from(zip(rand.choice(new_nodes, size=len(neigh), replace=True), neigh))

        n_nodes = [g.number_of_nodes() for g in graphs]
        max_n = max(n_nodes)
        X = torch.zeros(
            (batch_size, max_n),
            dtype=torch.int, device="cpu")
        E = torch.from_numpy(np.stack([
            np.pad(
                nx.to_numpy_array(g),
                (0, max_n - g.number_of_nodes()))
            for g in graphs]))

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list


    @torch.no_grad()
    def sample_batch_e2e_grad(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()

        for t_int in trange(0, self.cfg.sample.sample_steps):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / self.cfg.sample.sample_steps
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / self.cfg.sample.sample_steps

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt_grad(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Sample
        X, E, y = discrete_sampled_s.X, discrete_sampled_s.E, discrete_sampled_s.y

        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list


    @torch.no_grad()
    def sample_batch_rule_cond(
        self,
        batch_id: int,
        batch_size: int,
        save_final: int,
        num_nodes: Optional[int] = None,
        save_visualization: bool = True,
        **kwargs
    ):
        assert not self.conditional

        max_gen_size = int(torch.searchsorted(self.node_dist.prob.cumsum(0), 0.8))

        if num_nodes is None:
            num_nodes = max_gen_size

        n_nodes = num_nodes * torch.ones(
            batch_size, device=self.device, dtype=torch.int
        )

        size = min(num_nodes, max_gen_size)

        node_mask = torch.ones(
            (batch_size, size), device=self.device, dtype=torch.bool)

        # Sample noise -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()

        for t_int in trange(0, self.cfg.sample.sample_steps):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / self.cfg.sample.sample_steps
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / self.cfg.sample.sample_steps

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        X_fin, y_fin = discrete_sampled_s.X, discrete_sampled_s.y
        E_full = sampled_s.E
        ds_name = self.cfg.general.name
        rng = np.random.default_rng(self.cfg.train.seed)

        n_anc = self.cfg.sample.n_anc
        max_gen_size += n_anc

        if ds_name == "planar":
            sub_func = E_subgraphs_planar
        elif ds_name == "tree":
            sub_func = E_subgraphs_base
        elif ds_name == "sbm":
            sub_func = E_subgraphs_sbm
        else:
            raise NotImplementedError(f"Dataset {ds_name} not supported.")

        while X_fin.size(1) < num_nodes:
            size = n_anc + min(num_nodes - X_fin.size(1), max_gen_size - n_anc)

            node_mask = torch.ones(
                (batch_size, size), device=self.device, dtype=torch.bool)

            # Sample noise -- z has size (n_samples, n_nodes, n_features)
            z_T = flow_matching_utils.sample_discrete_feature_noise(
                limit_dist=self.limit_dist, node_mask=node_mask
            )
            X, E, y = z_T.X, z_T.E, z_T.y

            assert (E == torch.transpose(E, 1, 2)).all()

            E_imp, node_map = sub_func(E_full, n_anc, rng)
            E_imp = F.pad(E_imp, (0, 0, 0, size - n_anc, 0, size - n_anc))

            for t_int in trange(0, self.cfg.sample.sample_steps):
                # this state
                t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
                t_norm = t_array / self.cfg.sample.sample_steps
                if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                    # to avoid failure mode of absorbing transition, add epsilon
                    t_norm = t_norm + 1e-6
                # next state
                s_array = t_array + 1
                s_norm = s_array / self.cfg.sample.sample_steps

                # Distort time
                t_norm = self.time_distorter.sample_ft(
                    t_norm, self.cfg.sample.time_distortion
                )
                s_norm = self.time_distorter.sample_ft(
                    s_norm, self.cfg.sample.time_distortion
                )

                # Sample z_s
                sampled_s, discrete_sampled_s, _ = self.sample_p_zs_given_zt(
                    t_norm,
                    s_norm,
                    X,
                    E,
                    y,
                    node_mask,
                    E_imp
                )

                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # print((E[:, :n_anc, :n_anc].argmax(-1) - E_imp[:, :n_anc, :n_anc].argmax(-1)).abs().sum() / batch_size)
            X_fin = torch.concat([X_fin, discrete_sampled_s.X[:, n_anc:]], 1)
            E_full = E_stitch(E_full, E, node_map)

        X, y = X_fin, y_fin
        E = E_full.argmax(-1)
        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list


    def compute_step_probs(self, R_t_X, R_t_E, X_t, E_t, dt):
        step_probs_X = R_t_X * dt  # type: ignore # (B, D, S)
        step_probs_E = R_t_E * dt  # (B, D, S)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        # assert (E_t.argmax(-1) < 4).all()
        step_probs_X.scatter_(-1, X_t.argmax(-1)[:, :, None], 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1)[:, :, :, None], 0.0)

        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs_X.scatter_(
            -1,
            X_t.argmax(-1)[:, :, None],
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            E_t.argmax(-1)[:, :, :, None],
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        # step 2 - merge to the original formulation
        prob_X = step_probs_X.clone()
        prob_E = step_probs_E.clone()

        return prob_X, prob_E

    def sample_p_zs_given_zt(
        self,
        t,
        s,
        X_t,
        E_t,
        y_t,
        node_mask,
        E_imp=None,
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        # bs, n, dx = X_t.shape
        # _, _, _, de = E_t.shape
        dt = (s - t)[0]

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        guidance_weight = self.cfg.general.guidance_weight

        # guidance
        if E_imp is not None:
            E_imp_max = E_imp.amax(-1)
            E_imp_mask = E_imp_max.unsqueeze(-1)
            noisy_data["E_t"] = noisy_data["E_t"] * (1 - E_imp_mask) + E_imp

        extra_data = self.compute_extra_data(noisy_data)
        pred = self(noisy_data, extra_data, node_mask)
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        limit_x = self.limit_dist.X
        limit_e = self.limit_dist.E

        G_1_pred = pred_X, pred_E
        G_t = X_t, E_t

        R_t_X, R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(
            t,
            node_mask,
            G_t,
            G_1_pred,
        )

        if self.conditional:
            uncond_y = torch.ones_like(y_t, device=self.device) * -1
            noisy_data["y_t"] = uncond_y

            extra_data = self.compute_extra_data(noisy_data)
            pred = self(noisy_data, extra_data, node_mask)

            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

            G_1_pred = pred_X, pred_E

            R_t_X_uncond, R_t_E_uncond = self.rate_matrix_designer.compute_graph_rate_matrix(
                t,
                node_mask,
                G_t,
                G_1_pred,
            )

            R_t_X = torch.exp(
                torch.log(R_t_X_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_X + 1e-6) * guidance_weight
            )
            R_t_E = torch.exp(
                torch.log(R_t_E_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_E + 1e-6) * guidance_weight
            )

        # guidance
        if E_imp is not None:
            R_t_E **= guidance_weight

        prob_X, prob_E = self.compute_step_probs(
            R_t_X, R_t_E, X_t, E_t, dt
        )

        # replacement all-but-last step
        # if E_imp is not None:
        #     prob_E[:, :E_imp.size(1), :E_imp.size(1)] = E_imp

        if s[0] == 1.0:
            prob_X, prob_E = pred_X, pred_E

        # replacement all steps
        # if E_imp is not None:
        #     prob_E[:, :E_imp.size(1), :E_imp.size(1)] = E_imp

        sampled_s = flow_matching_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_x)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        out_discrete = out_discrete.mask(node_mask, collapse=True).type_as(y_t)

        E_prob = prob_E[..., 1:].sum(dim=-1)
        diag_mask = torch.eye(
            E_prob.size(-1), device=self.device, dtype=torch.bool
        ).unsqueeze(0)
        E_prob = E_prob.masked_fill(diag_mask, 0.)
        e_mask1 = node_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = node_mask.unsqueeze(1)  # bs, 1, n, 1
        E_prob[(e_mask1 * e_mask2).squeeze(-1) == 0] = 0

        return out_one_hot, out_discrete, E_prob


    def sample_p_zs_given_zt_grad(
        self,
        t,
        s,
        X_t,
        E_t,
        y_t,
        node_mask,
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, dx = X_t.shape
        # bs, _, _, de = E_t.shape
        dt = (s - t)[0]

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t.float().detach().requires_grad_(True),
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        extra_data = self.compute_extra_data(noisy_data)

        with torch.enable_grad():
            pred = self(noisy_data, extra_data, node_mask)
            pred_E: torch.Tensor = pred.E
            mask = ~torch.eye(n, device=self.device, dtype=torch.bool)
            pred_E_flat = pred_E[:, mask]
            diff = pred_E_flat[..., 1:].sum() - pred_E_flat[..., 0]
            loss = (1 - diff.tanh()**2).sum(1).mean()
            grad, *_ = torch.autograd.grad(loss, noisy_data["E_t"])

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E - self.cfg.general.guidance_weight * grad, -1)  # bs, n, n, d0

        limit_x = self.limit_dist.X
        limit_e = self.limit_dist.E

        G_1_pred = pred_X, pred_E
        G_t = X_t, E_t

        R_t_X, R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(
            t,
            node_mask,
            G_t,
            G_1_pred,
        )

        prob_X, prob_E = self.compute_step_probs(
            R_t_X, R_t_E, X_t, E_t, dt
        )

        if s[0] == 1.0:
            prob_X, prob_E = pred_X, pred_E

        sampled_s = flow_matching_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_x)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        out_discrete = out_discrete.mask(node_mask, collapse=True).type_as(y_t)

        E_prob = prob_E[..., 1:].sum(dim=-1)
        diag_mask = torch.eye(
            E_prob.size(-1), device=self.device, dtype=torch.bool
        ).unsqueeze(0)
        E_prob = E_prob.masked_fill(diag_mask, 0.)
        e_mask1 = node_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = node_mask.unsqueeze(1)  # bs, 1, n, 1
        E_prob[(e_mask1 * e_mask2).squeeze(-1) == 0] = 0

        return out_one_hot, out_discrete, E_prob


    def compute_extra_data(self, noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        extra_features = self.extra_features(noisy_data)

        # one additional category is added for the absorbing transition
        X, E, y = self.noise_dist.ignore_virtual_classes(
            noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        )
        noisy_data_to_mol_feat = noisy_data.copy()
        noisy_data_to_mol_feat["X_t"] = X
        noisy_data_to_mol_feat["E_t"] = E
        noisy_data_to_mol_feat["y_t"] = y
        extra_molecular_features = self.domain_features(noisy_data_to_mol_feat)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def search_hyperparameters(self):
        """
        Grid search for sampling hypeparameters.
        The num_step_list is tunable based on requirements.
        """

        num_step_list = [5, 10, 50, 100, 1000]
        if self.cfg.dataset.name == "qm9":
            num_step_list = [1, 5, 10, 50, 100, 500]
        if self.cfg.dataset.name in ["guacamol", 'moses', 'zinc']:  # accelerate
            num_step_list = [50]

        if self.cfg.sample.search == "all":
            results_df = self.search_distortion(num_step_list)
            results_df = self.search_stochasticity(num_step_list)
            results_df = self.search_target_guidance(num_step_list)
        elif self.cfg.sample.search == "distortion":
            results_df = self.search_distortion(num_step_list)
        elif self.cfg.sample.search == "stochasticity":
            results_df = self.search_stochasticity(num_step_list)
        elif self.cfg.sample.search == "target_guidance":
            results_df = self.search_target_guidance(num_step_list)
        else:
            raise NotImplementedError(
                f"Search type {self.cfg.sample.search} not implemented."
            )

        print("Finished searching. Results saved to search_hyperparameters.csv")

    def search_distortion(self, num_step_list):
        """
        Grid search for sampling distortion.
        """
        results_df = pd.DataFrame()
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        # distortion_list = ["identity", "polydec"]

        for num_step in num_step_list:
            for distortor in distortion_list:
                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.time_distortion = distortor
                print(
                    f"############# Testing num steps: {num_step}, distortor: {distortor} #############"
                )
                samples, labels = self.sample(
                    is_test=True,
                    save_samples=self.cfg.general.save_samples,
                    save_visualization=False,
                )
                res = self.evaluate_samples(
                    samples=samples, labels=labels, is_test=True
                )
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["distortor"] = distortor
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                # save at each step as well
                results_df.to_csv(f"search_distortion.csv")

        # set back to default value
        self.cfg.sample.time_distortion = "identity"

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "distortor"], inplace=True)
        results_df.to_csv(f"search_distortion.csv")

    def search_stochasticity(self, num_step_list):
        """
        Grid search for stochasticity level eta.
        The num_step_list is tunable based on requirements.
        """
        results_df = pd.DataFrame()
        eta_list = [0.0, 5, 10, 25, 50, 100, 200]
        # eta_list = [5, 10]
        for num_step in num_step_list:
            for eta in eta_list:
                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.eta = eta
                self.rate_matrix_designer.eta = eta
                print(
                    f"############# Testing num steps: {num_step}, eta: {eta} #############"
                )
                samples, labels = self.sample(
                    is_test=True,
                    save_samples=self.cfg.general.save_samples,
                    save_visualization=False,
                )
                res = self.evaluate_samples(
                    samples=samples, labels=labels, is_test=True
                )
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["eta"] = eta
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                # save at each step as well
                results_df.to_csv(f"search_stochasticity.csv")

        # set back to default value
        self.cfg.sample.eta = 0.0

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "eta"], inplace=True)
        results_df.to_csv(f"search_stochasticity.csv")

    def search_target_guidance(self, num_step_list):
        """
        Grid search for target guidance omega.
        The num_step_list is tunable based on requirements.
        """
        results_df = pd.DataFrame()
        omega_list = [
            0.0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            1.0,
            2.0,
        ]  # tunable based on requirements
        # omega_list = [0.5, 0.01]  # tunable based on requirements

        for num_step in num_step_list:
            for omega in omega_list:
                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.omega = omega
                self.rate_matrix_designer.omega = omega
                print(
                    f"############# Testing num steps: {num_step}, omega: {omega} #############"
                )
                samples, labels = self.sample(
                    is_test=True,
                    save_samples=self.cfg.general.save_samples,
                    save_visualization=False,
                )
                res = self.evaluate_samples(
                    samples=samples, labels=labels, is_test=True
                )
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["omega"] = omega
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                # save at each step as well
                results_df.to_csv(f"search_target_guidance.csv")

        # set back to default value
        self.cfg.sample.omega = 0.0

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "omega"], inplace=True)
        results_df.to_csv(f"search_target_guidance.csv")


# Adapted from: https://github.com/prodigy-diffusion/code/blob/main/project_bisection.py

def _to_chordal(adj: np.ndarray) -> np.ndarray:
    g_chordal, _ = nx.complete_to_chordal_graph(nx.from_numpy_array(adj))
    return nx.to_numpy_array(g_chordal)


def project(
        xs: torch.Tensor, adjs: torch.Tensor, constr: str,
        map_fn) -> tuple[torch.Tensor, torch.Tensor]:
    if constr == "chordal":
        adjs_chordal = tuple(map_fn(_to_chordal, (adj for adj in adjs.numpy(force=True))))

        return xs, torch.from_numpy(np.stack(adjs_chordal, axis=0)).to(device=adjs.device)

        # return xs, torch.from_numpy(
        #     np.stack(
        #         tuple(
        #             nx.to_numpy_array(nx.complete_to_chordal_graph(nx.from_numpy_array(adj))[0])
        #             for adj in adjs.numpy(force=True)), axis=0)).to(device=adjs.device)

    return xs, adjs


def drifted_project(
        xs: torch.Tensor, adjs: torch.Tensor,
        i: int, diff_steps: int, constr: str,
        sched_gamma: str, sched_params: list[float],
        exec: Optional[ProcessPoolExecutor]=None) -> tuple[torch.Tensor, torch.Tensor]:
    # xs:   sample_batch_size x max_num_nodes x node_attr_size
    # adjs: sample_batch_size x max_num_nodes x max_num_nodes
    map_fn = map if exec is None else exec.map

    if sched_gamma == "poly":
        # gamma_init: starting val from which to reach 1
        # i: next step from set {1, 2, ..., diff_steps}
        # time_pow: int > 1
        # drift is assumed to reach one at the end (i.e., when i / diff_steps = 1)
        gamma_init, time_pow = sched_params
        drift = (1 - gamma_init) * (i/diff_steps)**time_pow + gamma_init
        proj_xs, proj_adjs = project(xs, adjs, constr, map_fn)
        return xs + drift * (proj_xs - xs), adjs + drift * (proj_adjs - adjs)
    elif sched_gamma == 'exp-dist':
        proj_xs, proj_adjs = project(xs, adjs, constr, map_fn)
        adj_dist = (proj_adjs - adjs).reshape(adjs.shape[0], -1).norm(dim=1, p=2) / (adjs.shape[1] * adjs.shape[2])
        x_dist = (proj_xs - xs).reshape(xs.shape[0], -1).norm(dim=1, p=2) / (xs.shape[1] * xs.shape[2])
        thresh, beta = sched_params
        drift_x = torch.where(x_dist < thresh, torch.ones_like(x_dist), torch.exp(-(x_dist - thresh) * beta))
        drift_adj = torch.where(adj_dist < thresh, torch.ones_like(x_dist), torch.exp(-(adj_dist - thresh) * beta))
        return xs + drift_x[:, None, None] * (proj_xs - xs), adjs + drift_adj[:, None, None] * (proj_adjs - adjs)
    elif sched_gamma == "fixed":
        drift, *_ = sched_params
        proj_xs, proj_adjs = project(xs, adjs, constr, map_fn)
        return xs + drift * (proj_xs - xs), adjs + drift * (proj_adjs - adjs)

    return xs, adjs


def min_connected_graph(graph: nx.Graph) -> nx.Graph:
    G_connected = graph.copy()
    components = list(nx.connected_components(G_connected))
    rep_nodes = _, *rep_nodes_tail = [next(iter(c)) for c in components]
    G_connected.add_edges_from(zip(rep_nodes, rep_nodes_tail))

    return G_connected


def planar_separator_bfs(graph: nx.Graph) -> tuple[list[int], list[int], list[int]]:
    if not nx.is_connected(graph):
        raise ValueError("Graph must be connected.")

    current_node = next(iter(graph.nodes))
    best_layers = current_node
    max_depth = -1

    for _ in range(graph.number_of_nodes()):
            layers = list(nx.bfs_layers(graph, current_node))
            depth = len(layers)

            if depth > max_depth:
                max_depth = depth
                best_layers = layers

            current_node, *_ = layers[-1]

    threshold = graph.number_of_nodes() // 2
    cumulative_count = 0
    separator_level_index = -1

    for i, layer in enumerate(best_layers):
        if cumulative_count + len(layer) >= threshold:
            separator_level_index = i
            break

        cumulative_count += len(layer)

    part_a = sorted(set(n for lay in layers[:separator_level_index] for n in lay))
    separator = sorted(set(n for n in layers[separator_level_index]))
    part_b = sorted(set(n for lay in layers[separator_level_index+1:] for n in lay))

    return part_a, separator, part_b


def E_subgraphs_base(
        E: torch.Tensor,
        n_anc: int,
        rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    :param E: tensor (bs, n, n, d0)
    :param n_anc: int
    :param rand: random generator
    :return (E_sub, indices): tensor (bs, n_anc, n_anc, d0), tensor (bs, n_anc)
    """
    graphs = [
        nx.from_numpy_array(E_i)
        for E_i in E.argmax(-1).numpy(force=True)
    ]
    B, N, _, _ = E.shape

    start_nodes = rng.choice(N, size=B, replace=True)  # bs, n
    indices = torch.tensor(
        [
            [u, *islice((v for _, v in nx.bfs_edges(g, u)), n_anc - 1)]
            for g, u in zip(graphs, start_nodes)
        ], device=E.device)  # bs, n_anc

    batch_idx = torch.arange(B, device=E.device).view(B, 1, 1)
    row_idx = indices.view(B, n_anc, 1)
    col_idx = indices.view(B, 1, n_anc)

    return E[batch_idx, row_idx, col_idx], indices


def E_subgraphs_planar(
        E: torch.Tensor,
        n_anc: int,
        rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    :param E: tensor (bs, n, n, d0)
    :param n_anc: int
    :param rand: random generator
    :return (E_sub, indices): tensor (bs, n_anc, n_anc, d0), tensor (bs, n_anc)
    """
    graphs = [
        nx.from_numpy_array(E_i)
        for E_i in E.argmax(-1).numpy(force=True)
    ]
    B, N, _, _ = E.shape

    outer_faces: list[list[int]] = []

    for g in graphs:
        is_planar, embedding = nx.check_planarity(g)

        if not is_planar:
            u = rng.choice(N)
            outer_faces.append([u, *islice((v for _, v in nx.bfs_edges(g, u)), n_anc - 1)])

            continue

        faces = []
        visited_half_edges = set()

        for u in embedding.nodes():
            for v in embedding.neighbors(u):
                if (u, v) not in visited_half_edges:
                    face_nodes = embedding.traverse_face(u, v, mark_half_edges=visited_half_edges)

                faces.append(face_nodes)

        for face in faces:
            if len(face) == n_anc:
                outer_faces.append(face)
                break

    indices = torch.tensor(outer_faces, device=E.device)

    batch_idx = torch.arange(B, device=E.device).view(B, 1, 1)
    row_idx = indices.view(B, n_anc, 1)
    col_idx = indices.view(B, 1, n_anc)

    return E[batch_idx, row_idx, col_idx], indices


def E_subgraphs_sbm(
        E: torch.Tensor,
        n_anc: int,
        rng: np.random.Generator,
        refinement_steps=50) -> tuple[torch.Tensor, torch.Tensor]:
    """
    :param E: tensor (bs, n, n, d0)
    :param n_anc: int
    :param rand: random generator
    :param refinement_steps: int
    :return (E_sub, indices): tensor (bs, n_anc, n_anc, d0), tensor (bs, n_anc)
    """
    B, N, _, _ = E.shape
    indices = []

    for E_i in E.argmax(-1).numpy(force=True):
        gt_g = gt.Graph()
        gt_g.add_edge_list(np.argwhere(E_i))

        try:
            state = gt.minimize_blockmodel_dl(gt_g)
        except ValueError:
            g = nx.from_numpy_array(E_i)
            u = rng.choice(N)
            indices.append([u, *islice((v for _, v in nx.bfs_edges(g, u)), n_anc - 1)])

            continue

        for _ in range(refinement_steps):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

        b = state.get_blocks()
        b = gt.contiguous_map(state.get_blocks())
        state = state.copy(b=b)
        n_blocks = state.get_nonempty_B()

        block_labels = b.get_array()
        block_nodes: list[np.ndarray] = []

        for block_id in range(n_blocks):
            nodes, *_ = np.nonzero(block_labels == block_id)
            block_nodes.append(nodes)

        block_nodes.sort(key=lambda x: len(x))
        curr_inds = block_nodes[0].tolist()
        remaining_nodes_iter = iter(np.concatenate(block_nodes[1:]))

        while len(curr_inds) < n_anc:
            curr_n = next(remaining_nodes_iter)
            if E_i[curr_n][curr_inds].any():
                curr_inds.append(curr_n)

        indices.append(curr_inds[:n_anc])

    indices = torch.tensor(indices, device=E.device)

    batch_idx = torch.arange(B, device=E.device).view(B, 1, 1)
    row_idx = indices.view(B, n_anc, 1)
    col_idx = indices.view(B, 1, n_anc)

    return E[batch_idx, row_idx, col_idx], indices


def E_stitch(
        E_base: torch.Tensor, E_exp: torch.Tensor, node_map: torch.Tensor) -> torch.Tensor:
    """
    :param E: tensor (bs, n, n, d0)
    :param E: tensor (bs, n, n, d0)
    :param node_map: tensor (bs, n_anc)
    :return E: tensor (bs, 2 * n - n_anc, 2 * n - n_anc, d0)
    """
    B, N_base, _, _ = E_base.shape
    _, N_ext, _, _ = E_exp.shape
    _, K = node_map.shape
    n_new = N_ext - K
    M = N_base + n_new

    E = F.pad(E_base, (0, 0, 0, n_new, 0, n_new))
    E[:, N_base:, N_base:, :] = E_exp[:, K:, K:, :]

    batch_idx = torch.arange(B, device=E.device).view(B, 1, 1)
    cross_block = E_exp[:, :K, K:, :]

    idx_ov = node_map.view(B, K, 1)
    idx_new = torch.arange(N_base, M, device=E.device).view(1, 1, n_new)
    E[batch_idx, idx_ov, idx_new] = cross_block

    idx_new_T = idx_new.view(1, n_new, 1)
    idx_ov_T = idx_ov.view(B, 1, K)
    E[batch_idx, idx_new_T, idx_ov_T] = cross_block.transpose(1, 2)

    return E
