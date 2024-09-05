config_mnist = {
    "dataset": ["mnist"],
    "label_name": ["label"],
    "features_name": ["image"],
    "partition_by": ["label"],
    "split": ["train"],
    "shuffle": [True],
    "seed": list(range(42, 47))
}

config_cifar10 = {
    "dataset": ["cifar10"],
    "label_name": ["label"],
    "features_name": ["img"],
    "partition_by": ["label"],
    "split": ["train"],
    "shuffle": [True],
    "seed": list(range(42, 47))
}

config_cifar100 = {
    "dataset": ["cifar100"],
    "label_name": ["fine_label"],
    "features_name": ["img"],
    "partition_by": ["fine_label"],
    "split": ["train"],
    "shuffle": [True],
    "seed": list(range(42, 47))
}
# Configure FEMNIST as if it didn't have the writer_id
config_femnist_not_natural = {
    "dataset": ["flwrlabs/femnist"],
    "split": ["train"],
    "partition_by": ["character"],
    "label_name": ["character"],
    "features_name": ["image"],
    "shuffle": [True],
    "seed": list(range(42, 47))

}

# NaturalID datasets

config_femnist = {
    "dataset": ["flwrlabs/femnist"],
    "split": ["train"],
    "partition_by": ["writer_id"],
    "label_name": ["character"],
    "features_name": ["image"],
    "shuffle": [True],
    "seed": list(range(42, 47))

}

config_speech_commands = {
        "dataset": ["speech_commands"],
        "subset": ["v0.01"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["speaker_id"],
        "label_name": ["label"]   
}

config_shakespeare = {
        "dataset": ["flwrlabs/shakespeare"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["character_id"],
        "label_name": ["y"],
        "features_name": ["x"],

}

config_sentiment140 = {
        "dataset": ["sentiment140"],
        "split": ["train"],
        "shuffle": [False],
        "partition_by": ["user"],
        "label_name": ["sentiment"],
        "features_name": ["text"],

}

natural_datasets_param_grid = [
    config_femnist,
    #config_speech_commands,
    #config_shakespeare,
    #config_sentiment140
]

no_natural_datasets_param_grid = [
    config_cifar10, config_mnist, config_cifar100
]

# Add the natural_id datasets to the simulate the division on them
for natural_dataset_param_grid in natural_datasets_param_grid:
    fds_copy = natural_dataset_param_grid.copy()
    fds_copy["partition_by"] = fds_copy["label_name"]
    fds_copy["shuffle"] = [True]
    no_natural_datasets_param_grid.append(
        {**fds_copy, "seed": list(range(42, 47))})

all_fds_configs = [*no_natural_datasets_param_grid, *natural_datasets_param_grid]
fds_used_configs = [config_cifar10, config_cifar100, config_femnist, config_mnist]
