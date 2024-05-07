import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

from configs.run_configs import run_config
from configs.partitioner_configs import *
from configs.fds_configs import *
from configs.metrics_configs import *

if __name__ == "__main__":
    # have a run config: n_repeats etc

    # iterate over all configs/some other specification
    # save the results of each of the runs
    for fds_param_grid in natural_datasets_param_grid:
        fds_product = itertools.product(
            *(fds_param_grid[key] for key in fds_param_grid))
        fds_single_config_list = [
            dict(zip(fds_param_grid.keys(), values)) for values in fds_product]

        for single_fds in fds_single_config_list:
            print(single_fds)
            split = single_fds.pop("split")

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
                    elif partitioner_signature is NaturalIdPartitioner:
                        single_partitioner_config["partition_by"] = single_fds[
                            "partition_by"]
                    partitioner = partitioner_signature(**single_partitioner_config)
                    fds_kwargs = {**single_fds, "partitioners": {split: partitioner}}
                    if "partition_by" in fds_kwargs:
                        fds_kwargs.pop("partition_by")
                    fds = FederatedDataset(**fds_kwargs)
                    fds.load_partition(0)
                    # Different metrics calculation

                    for metric_config in metrics_configs:
                        metrics_fnc = metric_config["object"]
                        metrics_kwargs = {"partitioner": partitioner,
                                          **metric_config["kwargs"]}
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
                            metrics_avg_results[metrics_fnc.__name__] = {}
                        metrics_avg_results[metrics_fnc.__name__][
                            str(tuple(single_partitioner_config.items()))] = metric_avg

                save_results_dir_path = (f"../results/{single_fds['dataset']}/"
                                         f"{partitioner_signature.__name__}/shuffle_seed_{single_fds.get('seed', 'no_seed')}.json")
                save_results_dir_path = Path(save_results_dir_path)
                save_results_dir_path.parent.mkdir(parents=True, exist_ok=True)
                # pd.DataFrame(metrics_avg_results).to_csv(save_results_dir_path)
                with open(save_results_dir_path, "w") as file:
                    json.dump(metrics_avg_results, file, indent=4)
                print(f"Metrics saved in {save_results_dir_path}")
