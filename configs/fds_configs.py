natural_datasets_param_grid = [
    {
        "dataset": ["speech_commands"],
        "subset": ["v0.01"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["speaker_id"],
        "label_name": ["label"]

    },
    {
        "dataset": ["adamnarozniak/femnist"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["writer_id"],
        "label_name": ["character"]

    },

    {
        "dataset": ["sentiment140"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["user"],
        "label_name": ["sentiment"]

    },
    {
        "dataset": ["flwrlabs/shakespeare"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["character_id"],
        "label_name": ["y"]

    },
]

no_natural_datasets_param_grid = [
    {
        "dataset": ["cifar10", "mnist"],
        "label_name": ["label"],
        "split": ["train"],
        "shuffle": [True],
        "seed": list(range(42, 47))
    },
    {
        "dataset": ["cifar100"],
        "label_name": ["fine_label"],
        "split": ["train"],
        "shuffle": [True],
        "seed": list(range(42, 47))
    },
]
# Add the natural_id datasets to the simulate the division on them
for natural_dataset_param_grid in natural_datasets_param_grid:
    no_natural_datasets_param_grid.append(
        {**natural_dataset_param_grid, "seed": list(range(42, 47))})

fds_param_grids = [*no_natural_datasets_param_grid, *natural_datasets_param_grid]
