from pathlib import Path
from typing import Dict

from flwr_datasets import FederatedDataset

from configs.fds_configs import (
    natural_datasets_param_grid,
    no_natural_datasets_param_grid,
    config_cifar10,
)
from configs.metrics_configs import (
    all_metrics,
    metrics_earths_mover,
    metrics_hellinger,
    metrics_kl,
)
from configs.partitioner_configs import (
    natural_partitioner_configs,
    no_natural_partitioner_configs,
    config_iid_partitioner
)
from heterogeneity.config_utils import yeild_configs
import numpy as np
import pandas as pd

def run_heterogeneity_experiment(fds: FederatedDataset, fds_kwargs: Dict, partitioner_kwargs: Dict, metric_config: Dict, label_name: str):
    metrics_fnc = metric_config["object"]
    print(f"metric function: {metrics_fnc.__name__}")
    metric_config["kwargs"]["label_name"] = label_name
    metrics_kwargs = {
        "partitioner": fds.partitioners["train"],
        **metric_config["kwargs"],
    }
    print("metrics kwargs")
    print(metrics_kwargs)
    try:
        # trigger the assigment of the data
        _ = fds.load_partition(0)
        metric_list, metric_avg = metrics_fnc(**metrics_kwargs)
        if any(
            [
                metric_list_val in [np.inf, -np.inf]
                for metric_list_val in metric_list
            ]
        ):
            metric_list, metric_avg = np.nan, np.nan
    except ValueError as e:
        print(e)
        print(f"Sampling failed")
        metric_list, metric_avg = np.nan, np.nan

    print("Metric avg: ", metric_avg)
    # Save exmperiments results as quick as they are available
    # (then append as the new experiments come)
    save_results_dir_path = (
        f"results-test-heterogeneity-refactor-2024-03-09/{fds_kwargs['dataset']}/"
        f"{fds.partitioners['train'].__class__.__name__}/{metrics_fnc.__name__}.csv"
    )
    save_results_dir_path = Path(save_results_dir_path)
    include_header = True
    if save_results_dir_path.exists():
        include_header = False
    save_results_dir_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            [
                *partitioner_kwargs.values(),
                fds_kwargs.get("seed", "default"),
                metrics_fnc.__name__,
                metric_avg,
            ]
        ],
        columns=[
            *list(partitioner_kwargs.keys()),
            "fds_seed",
            "metric_name",
            "metric_value",
        ],
    ).to_csv(save_results_dir_path, index=False, mode="a", header=include_header)
    print(f"Saved results to {save_results_dir_path}")

if __name__ == "__main__":
    MODE: str = "CUSTOM"

    if MODE == "NATURAL-ID":
        print("Running NATURAL_ID")
        dataset_param_grid = natural_datasets_param_grid
        partitioner_param_grid = natural_partitioner_configs
    elif MODE == "NO-NATURAL-ID":
        print("Running NO_NATURAL_ID")
        dataset_param_grid = no_natural_datasets_param_grid
        partitioner_param_grid = no_natural_partitioner_configs
    elif MODE == "CUSTOM":
        print("Running CUSTOM")
        dataset_param_grid = [config_cifar10]
        partitioner_param_grid = [config_iid_partitioner]
    else:
        raise ValueError(f"Invalid MODE: {MODE}")

    METRIC_MODE: str = "HELLINGER"
    if METRIC_MODE == "HELLINGER":
        metrics_configs = [metrics_hellinger]
    elif METRIC_MODE == "KL":
        metrics_configs = [metrics_kl]
    elif METRIC_MODE == "EARTH_MOVER":
        metrics_configs = [metrics_earths_mover]
    elif METRIC_MODE == "ALL":
        metrics_configs = all_metrics
    else:
        raise ValueError(f"Invalid METRIC_MODE: {METRIC_MODE}")
    for (
        fds_kwargs,
        partitioner_kwargs,
        features_name,
        label_name,
    ) in yeild_configs(dataset_param_grid, partitioner_param_grid):
        fds = FederatedDataset(**fds_kwargs)
        for metrics_config in metrics_configs:
            print(
                f"Running Heterogeneity for {fds_kwargs['dataset']} with {fds_kwargs['partitioners']['train'].__class__.__name__}"
            )
            run_heterogeneity_experiment(
                fds, fds_kwargs, partitioner_kwargs, metrics_config, label_name
            )
