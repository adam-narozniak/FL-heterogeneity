from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, \
    ShardPartitioner, InnerDirichletPartitioner, NaturalIdPartitioner

from configs.class_constrained import ClassConstrainedPartitioner

config_iid_partitioner = {
    "object": IidPartitioner,
    "param_grid": {
        "num_partitions": [3, 10, 30, 100, 300, 1000]
    }
}

config_dirichlet_partitioner = {
    "object": DirichletPartitioner,
    "param_grid": {
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "alpha": [0.1, 0.3, 1., 3., 10., 100., 1000.],
        "partition_by": ["label"]
    }
}

config_natural_id_partitioner = {
    "object": NaturalIdPartitioner,
    "param_grid": {}
}

config_shard_partitioner = {
    "object": ShardPartitioner,
    "param_grid": {
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "num_shards_per_partition": [2, 3, 4, 5],
        "partition_by": ["label"]
    }
}

config_inner_dirichlet_partitioner = {
    "object": InnerDirichletPartitioner,
    "param_grid": {
        "alpha": [0.1, 0.3, 1., 3., 10., 100., 1000.],
        "sigma": [0.1, 0.3, 1., 3.],
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "partition_by": ["label"],
        "min_partition_size": 0,
    }
}

config_class_constrained = {
    "object": ClassConstrainedPartitioner,
    "param_grid": {
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "partition_by": ["label"],
        "num_classes_per_partition": [2, 3, 4, 5],#[int(0.2 * 62), int(0.3 * 62), int(0.4 * 62), int(0.5 * 62)],
        "first_class_deterministic_assignment": [True]
    }
}

config_class_constrained_full_random = {
    "object": ClassConstrainedPartitioner,
    "param_grid": {
        "num_partitions": [3, 10, 30, 100, 300, 1000],
        "partition_by": ["label"],
        "num_classes_per_partition": [2, 3, 4, 5],
        "first_class_deterministic_assignment": [True]
    }
}

natural_partitioner_configs = [config_natural_id_partitioner]

no_natural_partitioner_configs = [config_dirichlet_partitioner]#, config_shard_partitioner, ]
