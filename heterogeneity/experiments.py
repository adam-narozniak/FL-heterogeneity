import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset

from configs.fl_configs import fl_configs
from configs.fds_configs import no_natural_datasets_param_grid, natural_datasets_param_grid, config_femnist, config_mnist, config_cifar10, config_cifar100
from configs.metrics_configs import metrics_configs
from configs.partitioner_configs import natural_partitioner_configs, no_natural_partitioner_configs, config_pathological, config_iid_partitioner, config_dirichlet_partitioner
from heterogeneity.fl.fl_loop_fnc import create_dataloaders, get_net, run_fl_experiment
from heterogeneity.partitioners_utils import create_lognormal_partition_sizes
from flwr_datasets.partitioner import DirichletPartitioner, ShardPartitioner, NaturalIdPartitioner, InnerDirichletPartitioner, PathologicalPartitioner

def yeild_configs(datasets_param_grid, partitioner_param_grid):
    for fds_param_grid in datasets_param_grid:
        fds_product = itertools.product(
            *(fds_param_grid[key] for key in fds_param_grid))
        fds_single_config_list = [
            dict(zip(fds_param_grid.keys(), values)) for values in fds_product]
        for single_fds in fds_single_config_list:
            # print(single_fds)
            split = single_fds.pop("split")
            for partitioner_config in partitioner_param_grid:
                # print(partitioner_config)
                partitioner_signature = partitioner_config["object"]
                param_grid = partitioner_config["param_grid"]
                partitioner_config_product = itertools.product(
                    *(param_grid[key] for key in param_grid))

                # re-attach the keys to the values
                partitioner_kwargs_list = [dict(zip(param_grid.keys(), values))
                                                  for values in
                                                  partitioner_config_product]

                for partitioner_kwargs in partitioner_kwargs_list:
                    additional_to_save = None
                    if partitioner_signature in [DirichletPartitioner, ShardPartitioner, PathologicalPartitioner]:
                        partitioner_kwargs["seed"] = single_fds["seed"]
                        if "partition_by" in single_fds:
                            partitioner_kwargs["partition_by"] = single_fds["partition_by"]
                    elif partitioner_signature is NaturalIdPartitioner:
                        partitioner_kwargs["partition_by"] = single_fds[
                            "partition_by"]
                    if partitioner_signature is InnerDirichletPartitioner:
                        partitioner_kwargs["partition_sizes"] = create_lognormal_partition_sizes(single_fds["dataset"], partitioner_kwargs["num_partitions"], partitioner_kwargs["sigma"])
                        additional_to_save = {}
                        additional_to_save["sigma"] =  partitioner_kwargs.pop("sigma")
                        additional_to_save["num_partitions"] = partitioner_kwargs.pop("num_partitions")
                        if "partition_by" in single_fds:
                            partitioner_kwargs["partition_by"] = single_fds["partition_by"]
                        partitioner = partitioner_signature(**partitioner_kwargs)
                        partitioner_kwargs["sigma"] = additional_to_save["sigma"]
                        partitioner_kwargs["num_partitions"] = additional_to_save["num_partitions"]
                        partitioner_kwargs.pop("partition_sizes")
                    else:
                        partitioner = partitioner_signature(**partitioner_kwargs)
                    # print("partitioner config:")
                    # print(partitioner_kwargs)
                    fds_kwargs = {**single_fds, "partitioners": {split: partitioner}}
                    if "partition_by" in fds_kwargs:
                        fds_kwargs.pop("partition_by")
                    labels_name = fds_kwargs.pop("labels_name")
                    features_name = fds_kwargs.pop("features_name")

                    yield fds_kwargs, partitioner_kwargs, features_name, labels_name

def run_experiments_from_configs(datasets_param_grid, partitioner_param_grid, metrics_configs, run_heterogeneity, run_fl, fl_configs):
    for fds_kwargs, single_partitioner_config, features_name, labels_name in yeild_configs(datasets_param_grid, partitioner_param_grid):
        fds = FederatedDataset(**fds_kwargs)
        
        # Different metrics calculation
        if run_heterogeneity:
            for metric_config in metrics_configs:
                metrics_fnc = metric_config["object"]
                print("metric function:")
                print(metrics_fnc)
                if labels_name is not None:
                    metric_config["kwargs"]["labels_name"] = labels_name
                metrics_kwargs = {"partitioner": fds["partitioners"]["train"],
                                **metric_config["kwargs"]}
                print("metrics kwargs")
                print(metrics_kwargs)
                try:
                    # trigger the assigment of the data
                    _ = fds.load_partition(0)
                    metric_list, metric_avg = metrics_fnc(**metrics_kwargs)
                    if any([metric_list_val in [np.inf, -np.inf] for
                            metric_list_val in metric_list]):
                        metric_list, metric_avg = np.nan, np.nan
                except ValueError as e:
                    print(e)
                    print(f"Sampling failed")
                    metric_list, metric_avg = np.nan, np.nan

                print("Metric avg: ", metric_avg)
                # Save exmperiments results as quick as they are available
                # (then append as the new experiments come)
                save_results_dir_path = (f"results-2024-03-09/{fds_kwargs['dataset']}/"
                                                                f"{fds['partitioners']['train'].__name__}/{metrics_fnc.__name__}.csv")
                save_results_dir_path = Path(save_results_dir_path)
                include_header = True
                if save_results_dir_path.exists():
                    include_header = False
                save_results_dir_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([[*single_partitioner_config.values(), fds_kwargs.get("seed", "default"), metrics_fnc.__name__, metric_avg]], columns=[*list(single_partitioner_config.keys()), "fds_seed", "metric_name", "metric_value"]).to_csv(save_results_dir_path, index=False, mode="a", header=include_header)
        if run_fl:
            for fl_config in fl_configs:
                print(f"Running FL for {fds_kwargs['dataset']} with {fds_kwargs['partitioners']['train'].__name__}")
                try:
                    trainloaders, testloaders, centralized_dataloader = create_dataloaders(fds, features_name=features_name, labels_name=labels_name, seed=42)
                    dataset_name = fds.load_split("train").info.dataset_name
                    num_classes = len(fds.load_split("train").unique(labels_name))
                    net = get_net(dataset_name, num_classes)
                    n_comunication_rounds = fl_config["n_comunication_rounds"]
                    num_partitions = fds.partitioners["train"].num_partitions
                    n_clients_per_round_train = num_partitions if num_partitions <= 10 else int(fl_config["n_clients_per_round_train"] * len(trainloaders))
                    n_clients_per_round_eval =  num_partitions if num_partitions <= 10 else int(fl_config["n_clients_per_round_eval"] * len(testloaders))
                    early_stopping = fl_config["early_stopping"]
                    num_local_epochs = fl_config["num_local_epochs"]
                    fl_seed = fl_config["seed"]
                    metrics_train_list, metrics_eval_list, metrics_aggregated_train_list, metrics_aggregated_eval_list, test_res = run_fl_experiment(n_comunication_rounds, n_clients_per_round_train, n_clients_per_round_eval, trainloaders, testloaders, centralized_dataloader, net, num_local_epochs, features_name, labels_name, early_stopping, seed=fl_seed)
                except ValueError as e:
                    print(f"Failed to load partitions: {e}")
                    metrics_train_list, metrics_eval_list, metrics_aggregated_train_list, metrics_aggregated_eval_list, test_res = np.nan, np.nan, np.nan, np.nan, {"eval_loss": np.nan, "eval_acc": np.nan, "best_communication_round": np.nan}
                finally:
                    metrics_to_save = [metrics_train_list, metrics_eval_list, metrics_aggregated_train_list, metrics_aggregated_eval_list, test_res]
                    metrics_names = ["metrics_train_list", "metrics_eval_list", "metrics_aggregated_train_list", "metrics_aggregated_eval_list", "test_res"]
                    for metrics_name, metric_to_save in zip(metrics_names, metrics_to_save):
                        save_results_dir_path = (f"results-2024-09-03-local-epochs-5-rounds-500-test-pathological/{fds_kwargs['dataset']}/"
                                                                f"{fds_kwargs['partitioners']['train'].__name__}/{metrics_name}.csv")
                        
                        save_results_dir_path = Path(save_results_dir_path)
                        include_header = True
                        if save_results_dir_path.exists():
                            include_header = False
                        save_results_dir_path.parent.mkdir(parents=True, exist_ok=True)

                        if metrics_name == "test_res":
                            pd.DataFrame([[*single_partitioner_config.values(), fds_kwargs.get("seed", "default"), *list(metric_to_save.values())]], columns=[*list(single_partitioner_config.keys()), "fds_seed", *list(metric_to_save.keys())]).to_csv(save_results_dir_path, index=False, mode="a", header=include_header)
                        else:
                            pd.DataFrame([[*single_partitioner_config.values(), fds_kwargs.get("seed", "default"), metric_to_save]], columns=[*list(single_partitioner_config.keys()), "fds_seed", metrics_name]).to_csv(save_results_dir_path, index=False, mode="a", header=include_header)
                        print(f"FL {metrics_name} saved")

                        
                    # pd.DataFrame([[*single_partitioner_config.values(), fds_kwargs.get("seed", "default"), metrics_eval_list]], columns=[*list(single_partitioner_config.keys()), "fds_seed", "metrics_train_list"]).to_csv(save_results_dir_path, index=False, mode="a", header=include_header)
                    # print("FL Training history saved")


                    # pd.DataFrame([[dataset_name, metrics_eval_list]], columns=["dataset_name", "metrics_eval_list"]).to_csv("./fl_eval_metrics.csv")
                    # print("FL Evaluation history saved")

                    # print("FL Training history aggregated:")
                    # print(metrics_aggregated_train_list)
                    # pd.DataFrame([[dataset_name, metrics_aggregated_train_list]], columns=["dataset_name", "metrics_aggregated_train_list"]).to_csv("./fl_train_aggregated_metrics.csv")

                    # print("FL Evaluation history aggregated:")
                    # print(metrics_aggregated_eval_list)
                    # pd.DataFrame([[dataset_name, metrics_aggregated_eval_list]], columns=["dataset_name", "metrics_aggregated_eval_list"]).to_csv("./fl_eval_aggregated_metrics.csv")

                    # pd.DataFrame([[*single_partitioner_config.values(), single_fds.get("seed", "default"), metrics_fnc.__name__, metric_avg]], columns=[*list(single_partitioner_config.keys()), "fds_seed", "metric_name", "metric_value"]).to_csv(save_results_dir_path, index=False, mode="a", header=include_header)


        

            # if metrics_fnc.__name__ not in metrics_avg_results:
            #     metrics_avg_results[metrics_fnc.__name__] = pd.DataFrame([], columns=[*list(single_partitioner_config.keys()), "fds_seed", "metric_name", "metric_value"])
            #     # metrics_avg_results[metrics_fnc.__name__]["fds_seed"] = single_fds.get("seed", "default")
                # metrics_avg_results[metrics_fnc.__name__]["metric_name"] = metrics_fnc.__name__
                # metrics_avg_results[metrics_fnc.__name__][
                #     "metric_value"] = metric_avg

            # metrics_avg_results[metrics_fnc.__name__][
            #     str(tuple(single_partitioner_config.items()))] = metric_avg
            # metrics_avg_results[metrics_fnc.__name__].loc[len(metrics_avg_results[metrics_fnc.__name__])] =  [*single_partitioner_config.values(), single_fds.get("seed", "default"), metrics_fnc.__name__, metric_avg]

    # Old way of saving the results (not easy for aggregation)
    # save_results_dir_path = (f"../results/{single_fds['dataset']}/"
    #                          f"{partitioner_signature.__name__}/shuffle_seed_{single_fds.get('seed', 'no_seed')}.json")
    # save_results_dir_path = Path(save_results_dir_path)
    # save_results_dir_path.parent.mkdir(parents=True, exist_ok=True)
    # # pd.DataFrame(metrics_avg_results).to_csv(save_results_dir_path)
    # with open(save_results_dir_path, "w") as file:
    #     json.dump(metrics_avg_results, file, indent=4)
    # print(f"Metrics saved in {save_results_dir_path}")
    
    # Save the heterogeneity metrics results (from a single partitioner [different configurations])
    # for name, value in metrics_avg_results.items():
    #     save_results_dir_path = (f"../results-new/{single_fds['dataset']}/"
    #                                                       f"{partitioner_signature.__name__}/{name}.csv")
    #     save_results_dir_path = Path(save_results_dir_path)
    #     include_header = True
    #     if save_results_dir_path.exists():
    #         include_header = False
    #     save_results_dir_path.parent.mkdir(parents=True, exist_ok=True)
    #     value.to_csv(save_results_dir_path, index=False, mode="a", header=include_header)

                

if __name__ == "__main__":

    MODE: str = "FL-EXPERIMENTS"
    if MODE == "NATURAL_ID":
        print("Running NATURAL_ID")
        dataset_param_grid = natural_datasets_param_grid
        partitioner_param_grid = natural_partitioner_configs
    elif MODE == "NO_NATURAL_ID":
        print("Running NO_NATURAL_ID")
        dataset_param_grid = no_natural_datasets_param_grid
        partitioner_param_grid = no_natural_partitioner_configs
    elif MODE == "FL-EXPERIMENTS":
        run_fl = True
        run_heterogeneity = False
        dataset_param_grid = [config_cifar10, config_cifar100, config_mnist]
        # Single seed
        for dataset_param in dataset_param_grid:
            dataset_param["seed"] = [42]
        partitioner_param_grid = [config_iid_partitioner]#config_iid_partitioner, config_dirichlet_partitioner]config_pathological
    else:
        raise ValueError("incorrect mode name")

    run_experiments_from_configs(
        datasets_param_grid=dataset_param_grid,
        partitioner_param_grid=partitioner_param_grid,
        metrics_configs=metrics_configs,
        run_heterogeneity=False,
        run_fl=True,
        fl_configs=fl_configs,
    )

