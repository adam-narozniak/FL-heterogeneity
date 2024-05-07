natural_datasets_param_grid = [
    {
        "dataset": ["speech_commands"],
        "subset": ["v0.01"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["speaker_id"]

    },
    {
        "dataset": ["adamnarozniak/femnist"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["writer_id"]

    },

    {
        "dataset": ["sentiment140"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["user"]

    },
    {
        "dataset": ["flwrlabs/shakespeare"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["character_id"]

    },
]

no_natural_datasets_param_grid = [
    {
        "dataset": ["cifar10", "cifar100", "mnist"],
        "split": ["train"],
        "shuffle": [True],
        "seed": list(range(42, 47))
    },
]
for natural_dataset_param_grid in natural_datasets_param_grid:
    no_natural_datasets_param_grid.append(
        {**natural_dataset_param_grid, "seed": list(range(42, 47))})

fds_param_grids = [*no_natural_datasets_param_grid, *natural_datasets_param_grid]
