from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    InnerDirichletPartitioner,
    NaturalIdPartitioner,
    ShardPartitioner,
)

from heterogeneity.fl.pathological_partitioner import PathologicalPartitioner

config_iid_partitioner = {
    "object": IidPartitioner,
    "param_grid": {"num_partitions": [3, 10, 30, 100, 300, 1000]},
}

config_dirichlet_partitioner = {
    "object": DirichletPartitioner,
    "param_grid": {
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0],
        "self_balancing": [False],
    },
}

config_natural_id_partitioner = {"object": NaturalIdPartitioner, "param_grid": {}}

config_shard_partitioner = {
    "object": ShardPartitioner,
    "param_grid": {
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "num_shards_per_partition": [2, 3, 4, 5],
    },
}

config_inner_dirichlet_partitioner = {
    "object": InnerDirichletPartitioner,
    "param_grid": {
        "alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0],
        "sigma": [0.1, 0.3, 1.0, 3.0],
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "min_partition_size": 0,
    },
}

config_pathological = {
    "object": PathologicalPartitioner,
    "param_grid": {
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "num_classes_per_partition": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "class_assignment_mode": ["deterministic", "random"],
    },
}

config_pathological_deterministic = {
    "object": PathologicalPartitioner,
    "param_grid": {
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "num_classes_per_partition": [0.2, 0.3, 0.4, 0.5],
        "class_assignment_mode": ["deterministic"],
    },
}

config_pathological_random = {
    "object": PathologicalPartitioner,
    "param_grid": {
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "num_classes_per_partition": [0.2, 0.3, 0.4, 0.5],
        "class_assignment_mode": ["random"],
    },
}

natural_partitioner_configs = [config_natural_id_partitioner]

no_natural_partitioner_configs = [
    config_dirichlet_partitioner
]  # , config_shard_partitioner, ]
