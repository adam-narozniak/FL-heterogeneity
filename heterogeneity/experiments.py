import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset

from configs.fds_configs import *
from configs.metrics_configs import *
from configs.partitioner_configs import *


def run_experiments_from_configs(natural_id_run: bool = False):
    # iterate over all configs/some other specification
    # save the results of each of the runs

    datasets_param_grid = natural_datasets_param_grid if natural_id_run else (
        no_natural_datasets_param_grid)

    for fds_param_grid in datasets_param_grid:
        fds_product = itertools.product(
            *(fds_param_grid[key] for key in fds_param_grid))
        fds_single_config_list = [
            dict(zip(fds_param_grid.keys(), values)) for values in fds_product]

        for single_fds in fds_single_config_list:
            print(single_fds)
            split = single_fds.pop("split")

            partitioner_configs = natural_partitioner_configs if natural_id_run else (
                no_natural_partitioner_configs)

            for partitioner_config in partitioner_configs:
                print(partitioner_config)
                partitioner_signature = partitioner_config["object"]
                param_grid = partitioner_config["param_grid"]
                partitioner_config_product = itertools.product(
                    *(param_grid[key] for key in param_grid))

                # re-attach the keys to the values
                single_partitioner_config_list = [dict(zip(param_grid.keys(), values))
                                                  for values in
                                                  partitioner_config_product]

                metrics_avg_results = {}
                metrics_list_results = {}

                for single_partitioner_config in single_partitioner_config_list:
                    print(single_partitioner_config)
                    if partitioner_signature is DirichletPartitioner:
                        single_partitioner_config["seed"] = single_fds["seed"]
                        if "label_name" in single_fds:
                            single_partitioner_config["partition_by"] = single_fds["label_name"]
                    elif partitioner_signature is NaturalIdPartitioner:
                        single_partitioner_config["partition_by"] = single_fds[
                            "partition_by"]
                    partitioner = partitioner_signature(**single_partitioner_config)
                    fds_kwargs = {**single_fds, "partitioners": {split: partitioner}}
                    if "partition_by" in fds_kwargs:
                        fds_kwargs.pop("partition_by")
                    label_name = None
                    if "label_name" in fds_kwargs:
                        label_name = fds_kwargs.pop("label_name")
                    fds = FederatedDataset(**fds_kwargs)
                    fds.load_partition(0)
                    # Different metrics calculation

                    for metric_config in metrics_configs:
                        metrics_fnc = metric_config["object"]
                        if label_name is not None:
                            metric_config["kwargs"]["label_name"] = label_name
                        metrics_kwargs = {"partitioner": partitioner,
                                          **metric_config["kwargs"]}
                        print(metrics_kwargs)
                        try:

                            metric_list, metric_avg = metrics_fnc(**metrics_kwargs)
                            if any([metric_list_val in [np.inf, -np.inf] for
                                    metric_list_val in metric_list]):
                                metric_list, metric_avg = np.nan, np.nan
                        except ValueError:
                            print(f"Sampling failed")
                            metric_list, metric_avg = np.nan, np.nan

                        print(metric_avg)
                        if metrics_fnc.__name__ not in metrics_avg_results:
                            metrics_avg_results[metrics_fnc.__name__] = pd.DataFrame(single_partitioner_config, index=[0])
                            metrics_avg_results[metrics_fnc.__name__]["fds_seed"] = single_fds.get("seed", "default")
                            metrics_avg_results[metrics_fnc.__name__]["metric_name"] = metrics_fnc.__name__
                            metrics_avg_results[metrics_fnc.__name__][
                                "metric_value"] = metric_avg

                        # metrics_avg_results[metrics_fnc.__name__][
                        #     str(tuple(single_partitioner_config.items()))] = metric_avg
                        metrics_avg_results[metrics_fnc.__name__].loc[len(metrics_avg_results[metrics_fnc.__name__])] =  [*single_partitioner_config.values(), single_fds.get("seed", "default"), metrics_fnc.__name__, metric_avg]

                # save_results_dir_path = (f"../results/{single_fds['dataset']}/"
                #                          f"{partitioner_signature.__name__}/shuffle_seed_{single_fds.get('seed', 'no_seed')}.json")
                # save_results_dir_path = Path(save_results_dir_path)
                # save_results_dir_path.parent.mkdir(parents=True, exist_ok=True)
                # # pd.DataFrame(metrics_avg_results).to_csv(save_results_dir_path)
                # with open(save_results_dir_path, "w") as file:
                #     json.dump(metrics_avg_results, file, indent=4)
                # print(f"Metrics saved in {save_results_dir_path}")
                for name, value in metrics_avg_results.items():
                    save_results_dir_path = (f"../results/{single_fds['dataset']}/"
                                                                      f"{partitioner_signature.__name__}/{name}.csv")
                    save_results_dir_path = Path(save_results_dir_path)
                    include_header = True
                    if save_results_dir_path.exists():
                        include_header = False
                    save_results_dir_path.parent.mkdir(parents=True, exist_ok=True)
                    value.to_csv(save_results_dir_path, index=False, mode="a", header=include_header)


if __name__ == "__main__":
    # print("Non natural id running")
    # run_experiments_from_configs(natural_id_run=False)
    print("Natural id partitioning running")
    run_experiments_from_configs(natural_id_run=True)

