import itertools

from flwr_datasets.partitioner import (
    DirichletPartitioner,
    InnerDirichletPartitioner,
    NaturalIdPartitioner,
    PathologicalPartitioner,
    ShardPartitioner,
)

from heterogeneity.partitioners_utils import create_lognormal_partition_sizes


def yeild_configs(datasets_param_grid, partitioner_param_grid):
    for fds_param_grid in datasets_param_grid:
        fds_product = itertools.product(
            *(fds_param_grid[key] for key in fds_param_grid)
        )
        fds_single_config_list = [
            dict(zip(fds_param_grid.keys(), values)) for values in fds_product
        ]
        for single_fds in fds_single_config_list:
            # print(single_fds)
            split = single_fds.pop("split")
            for partitioner_config in partitioner_param_grid:
                # print(partitioner_config)
                partitioner_signature = partitioner_config["object"]
                param_grid = partitioner_config["param_grid"]
                partitioner_config_product = itertools.product(
                    *(param_grid[key] for key in param_grid)
                )

                # re-attach the keys to the values
                partitioner_kwargs_list = [
                    dict(zip(param_grid.keys(), values))
                    for values in partitioner_config_product
                ]

                for partitioner_kwargs in partitioner_kwargs_list:
                    additional_to_save = None
                    if partitioner_signature in [
                        DirichletPartitioner,
                        ShardPartitioner,
                        PathologicalPartitioner,
                    ]:
                        partitioner_kwargs["seed"] = single_fds["seed"]
                        if "partition_by" in single_fds:
                            partitioner_kwargs["partition_by"] = single_fds[
                                "partition_by"
                            ]
                    elif partitioner_signature is NaturalIdPartitioner:
                        partitioner_kwargs["partition_by"] = single_fds["partition_by"]
                    if partitioner_signature is InnerDirichletPartitioner:
                        partitioner_kwargs["partition_sizes"] = (
                            create_lognormal_partition_sizes(
                                single_fds["dataset"],
                                partitioner_kwargs["num_partitions"],
                                partitioner_kwargs["sigma"],
                            )
                        )
                        additional_to_save = {}
                        additional_to_save["sigma"] = partitioner_kwargs.pop("sigma")
                        additional_to_save["num_partitions"] = partitioner_kwargs.pop(
                            "num_partitions"
                        )
                        if "partition_by" in single_fds:
                            partitioner_kwargs["partition_by"] = single_fds[
                                "partition_by"
                            ]
                        partitioner = partitioner_signature(**partitioner_kwargs)
                        partitioner_kwargs["sigma"] = additional_to_save["sigma"]
                        partitioner_kwargs["num_partitions"] = additional_to_save[
                            "num_partitions"
                        ]
                        partitioner_kwargs.pop("partition_sizes")
                    else:
                        partitioner = partitioner_signature(**partitioner_kwargs)
                    # print("partitioner config:")
                    # print(partitioner_kwargs)
                    fds_kwargs = {**single_fds, "partitioners": {split: partitioner}}
                    if "partition_by" in fds_kwargs:
                        fds_kwargs.pop("partition_by")
                    label_name = fds_kwargs.pop("label_name")
                    features_name = fds_kwargs.pop("features_name")

                    yield fds_kwargs, partitioner_kwargs, features_name, label_name
