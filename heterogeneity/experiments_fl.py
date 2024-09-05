from pathlib import Path
from pprint import pprint
from typing import Dict

import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset
import torch

from configs.fds_configs import (
    config_cifar10,
    config_cifar100,
    config_femnist,
    config_mnist,
    natural_datasets_param_grid,
    no_natural_datasets_param_grid,
)
from configs.fl_configs import fl_configs
from configs.partitioner_configs import (
    config_dirichlet_partitioner,
    config_iid_partitioner,
    config_pathological,
    natural_partitioner_configs,
    no_natural_partitioner_configs,
)
from heterogeneity.config_utils import yeild_configs
from heterogeneity.fl.data import create_dataloaders
from heterogeneity.fl.fl_loop_fnc import run_fl_experiment
from heterogeneity.fl.model import get_net


def run_fl(
    fds: FederatedDataset,
    fds_kwargs: Dict,
    partitioner_kwargs: Dict,
    fl_config: Dict,
    label_name: str,
):
    # seed in torch for model weights init + dataloader shuffling
    seed = fl_config["seed"]
    # same seed in numpy for clients selection for each communication round 
    # (train and test clients are selected seperately from the same rng)
    torch.manual_seed(seed) 
    try:
        trainloaders, testloaders, centralized_dataloader = create_dataloaders(
            fds,
            features_name=features_name,
            label_name=label_name,
            seed=42,
        )
        dataset_name = fds.load_split("train").info.dataset_name
        num_classes = len(fds.load_split("train").unique(label_name))
        net = get_net(dataset_name, num_classes)
        n_comunication_rounds = fl_config["n_comunication_rounds"]
        num_partitions = fds.partitioners["train"].num_partitions
        n_clients_per_round_train = (
            num_partitions
            if num_partitions <= 10
            else int(fl_config["n_clients_per_round_train"] * len(trainloaders))
        )
        n_clients_per_round_eval = (
            num_partitions
            if num_partitions <= 10
            else int(fl_config["n_clients_per_round_eval"] * len(testloaders))
        )
        early_stopping = fl_config["early_stopping"]
        num_local_epochs = fl_config["num_local_epochs"]
        fl_seed = fl_config["seed"]
        (
            _,
            metrics_train_list,
            metrics_eval_list,
            metrics_aggregated_train_list,
            metrics_aggregated_eval_list,
            test_res,
        ) = run_fl_experiment(
            n_comunication_rounds,
            n_clients_per_round_train,
            n_clients_per_round_eval,
            trainloaders,
            testloaders,
            centralized_dataloader,
            net,
            num_local_epochs,
            features_name,
            label_name,
            early_stopping,
            seed=fl_seed,
        )
    except ValueError as e:
        print(f"Failed to load partitions: {e}")
        metrics_train_list = np.nan
        metrics_eval_list = np.nan
        metrics_aggregated_train_list = np.nan
        metrics_aggregated_eval_list = np.nan
        test_res = {
            "eval_loss": np.nan,
            "eval_acc": np.nan,
            "best_communication_round": np.nan,
        }
    finally:
        metrics_to_save = [
            metrics_train_list,
            metrics_eval_list,
            metrics_aggregated_train_list,
            metrics_aggregated_eval_list,
            test_res,
        ]
        metrics_names = [
            "metrics_train_list",
            "metrics_eval_list",
            "metrics_aggregated_train_list",
            "metrics_aggregated_eval_list",
            "test_res",
        ]
        for metrics_name, metric_to_save in zip(metrics_names, metrics_to_save):
            save_results_dir_path = (
                f"results-iid-adam/{fds_kwargs['dataset']}/"
                f"{fds_kwargs['partitioners']['train'].__class__.__name__}/{metrics_name}.csv"
            )

            save_results_dir_path = Path(save_results_dir_path)
            include_header = True
            if save_results_dir_path.exists():
                include_header = False
            save_results_dir_path.parent.mkdir(parents=True, exist_ok=True)

            if metrics_name == "test_res":
                pd.DataFrame(
                    [
                        [
                            *partitioner_kwargs.values(),
                            fds_kwargs.get("seed", "default"),
                            *list(metric_to_save.values()),
                        ]
                    ],
                    columns=[
                        *list(partitioner_kwargs.keys()),
                        "fds_seed",
                        *list(metric_to_save.keys()),
                    ],
                ).to_csv(
                    save_results_dir_path,
                    index=False,
                    mode="a",
                    header=include_header,
                )
            else:
                pd.DataFrame(
                    [
                        [
                            *partitioner_kwargs.values(),
                            fds_kwargs.get("seed", "default"),
                            metric_to_save,
                        ]
                    ],
                    columns=[
                        *list(partitioner_kwargs.keys()),
                        "fds_seed",
                        metrics_name,
                    ],
                ).to_csv(
                    save_results_dir_path,
                    index=False,
                    mode="a",
                    header=include_header,
                )
            print(f"FL {metrics_name} saved")


if __name__ == "__main__":

    save_dir_name = "results-iid-adam"

    MODE: str = "CUSTOM"
    if MODE == "NATURAL_ID":
        print("Running NATURAL_ID")
        dataset_param_grid = natural_datasets_param_grid
        partitioner_param_grid = natural_partitioner_configs
    elif MODE == "NO_NATURAL_ID":
        print("Running NO_NATURAL_ID")
        dataset_param_grid = no_natural_datasets_param_grid
        partitioner_param_grid = no_natural_partitioner_configs
    elif MODE == "CUSTOM":
        print("Running CUSTOM")
        dataset_param_grid = [config_mnist, config_cifar10, config_cifar100]
        partitioner_param_grid = [
            config_iid_partitioner,
            # config_dirichlet_partitioner,
            # config_pathological,
        ]
    else:
        raise ValueError(f"Invalid mode: {MODE}")
    # Single seed
    for dataset_param in dataset_param_grid:
        dataset_param["seed"] = [42]

    for (
        fds_kwargs,
        partitioner_kwargs,
        features_name,
        label_name,
    ) in yeild_configs(dataset_param_grid, partitioner_param_grid):
        fds = FederatedDataset(**fds_kwargs)
        for fl_config in fl_configs:
            print(
                f"Running Heterogeneity for {fds_kwargs['dataset']} with {fds_kwargs['partitioners']['train'].__class__.__name__}"
            )
            print("FL config:")
            pprint(fl_config)
            run_fl(
                fds, fds_kwargs, partitioner_kwargs, fl_config, label_name
            )
