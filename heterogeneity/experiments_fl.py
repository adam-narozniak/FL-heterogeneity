import argparse
from pathlib import Path
from pprint import pprint
from typing import Dict

import numpy as np
import torch
import wandb
from flwr_datasets import FederatedDataset
from sklearn.model_selection import ParameterGrid

from configs.fds_configs import (
    config_cifar10,
    config_cifar100,
    config_femnist_not_natural,
    config_mnist,
    natural_datasets_param_grid,
    no_natural_datasets_param_grid,
)
from configs.fl_configs import fl_configs
from configs.optimizer_configs import adam_config, optimizer_configs
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
from heterogeneity.fl.save_utils import save_fl_results
from heterogeneity.fds_utils import create_fds


def run_fl(
    fds: FederatedDataset,
    fds_kwargs: Dict,
    partitioner_kwargs: Dict,
    fl_config: Dict,
    optimizer_class,
    optimier_kwargs,
    features_name: str,
    label_name: str,
    train_frac: float,
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
            train_frac=train_frac,
            seed=fds_kwargs["seed"],
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
            optimizer_class,
            optimier_kwargs,
            num_local_epochs,
            features_name,
            label_name,
            early_stopping,
            seed=fl_seed,
        )
    except ValueError as e:
        print(f"Probably failed to load partitions.\n{e}")
        metrics_train_list = np.nan
        metrics_eval_list = np.nan
        metrics_aggregated_train_list = np.nan
        metrics_aggregated_eval_list = np.nan
        test_res = {
            "eval/loss": np.nan,
            "eval/acc": np.nan,
            "best_communication_round": np.nan,
        }
    return (
        metrics_train_list,
        metrics_eval_list,
        metrics_aggregated_train_list,
        metrics_aggregated_eval_list,
        test_res,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--experiment-name",
        type=str,
        default="testing",
        help="Name of the experiment",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    experiment_name: str = args.experiment_name
    results_directory_name = Path(f"results/{experiment_name}")
    results_directory_name.mkdir(parents=True, exist_ok=True)

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
        dataset_param_grid = [
            # config_femnist_not_natural,
            config_mnist,
            config_cifar10,
            config_cifar100, 
        ]  # , ]
        partitioner_param_grid = [
            # config_iid_partitioner,
            # config_dirichlet_partitioner,
            config_pathological,
        ]
        optimizer_configs_to_be_grid = [adam_config]
    else:
        raise ValueError(f"Invalid mode: {MODE}")
    # Single seed
    for dataset_param in dataset_param_grid:
        dataset_param["seed"] = [42]

    # Save all the configs before starting the experiments
    for idx, dataset_param_grid_single in enumerate(dataset_param_grid):
        with open(results_directory_name / "dataset_configs.txt", "a") as f:
            f.write(f"[{idx+1}/{len(dataset_param_grid)}]\n")
            f.write(f"{dataset_param_grid_single}\n")
    for idx, param_grid_single in enumerate(partitioner_param_grid):
        with open(results_directory_name / "partitioner_configs.txt", "a") as f:
            f.write(f"[{idx+1}/{len(partitioner_param_grid)}]\n")
            f.write(f"{param_grid_single}\n")
    for idx, fl_grid_single in enumerate(fl_configs):
        with open(results_directory_name / "fl_configs.txt", "a") as f:
            f.write(f"[{idx+1}/{len(fl_configs)}]\n")
            f.write(f"{fl_grid_single}\n")
    for idx, optim_grid_single in enumerate(optimizer_configs):
        with open(results_directory_name / "optim_configs.txt", "a") as f:
            f.write(f"[{idx+1}/{len(optimizer_configs)}]\n")
            f.write(f"{optim_grid_single}\n")

    for (
        fds_kwargs,
        partitioner_kwargs,
        features_name,
        label_name,
        train_frac,
    ) in yeild_configs(dataset_param_grid, partitioner_param_grid):
        fds = create_fds(fds_kwargs)
        optimizer_configs_grid = ParameterGrid(optimizer_configs_to_be_grid)
        for optimizer in optimizer_configs_grid:
            print("Optimizer config:")
            pprint(optimizer)
            optimizer_class = optimizer.pop("object")
            optimizer_kwargs = optimizer

            for fl_config in fl_configs:
                wandb.init(
                    project="fl-heterogeneity",
                    name=f"{experiment_name}/{fds_kwargs['dataset']}/"
                    f"{fds_kwargs['partitioners']['train'].__class__.__name__}({list(partitioner_kwargs.values())})/{optimizer_class.__name__}({optimizer_kwargs})",
                    group=f"{fds_kwargs['dataset']}",
                    tags=[f"{fds_kwargs['partitioners']['train'].__class__.__name__}", f"{optimizer_class.__name__}"],
                )
                
                wandb.config.update({"fds": fds_kwargs, "partitioner": partitioner_kwargs, "fl_config": fl_config, "optimizer": optimizer})
                print(
                    f"Running Heterogeneity for {fds_kwargs['dataset']} with {fds_kwargs['partitioners']['train'].__class__.__name__}"
                )
                print("FL config:")
                pprint(fl_config)

                (
                    metrics_train_list,
                    metrics_eval_list,
                    metrics_aggregated_train_list,
                    metrics_aggregated_eval_list,
                    test_res,
                ) = run_fl(
                    fds,
                    fds_kwargs,
                    partitioner_kwargs,
                    fl_config,
                    optimizer_class,
                    optimizer_kwargs,
                    features_name,
                    label_name,
                    train_frac,
                )
                experiment_save_path = f"{results_directory_name}/{fds_kwargs['dataset']}/"\
                    f"{fds_kwargs['partitioners']['train'].__class__.__name__}/"\
                    f"{optimizer_class.__name__+str(optimizer_kwargs).replace(' ', '')}"
                save_fl_results(
                    fds_kwargs,
                    partitioner_kwargs,
                    optimizer_kwargs,
                    experiment_save_path,
                    metrics_train_list,
                    metrics_eval_list,
                    metrics_aggregated_train_list,
                    metrics_aggregated_eval_list,
                    test_res,
                )
                wandb.finish()
