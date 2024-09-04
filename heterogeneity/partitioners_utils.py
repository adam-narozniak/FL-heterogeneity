from datasets import load_dataset_builder
import numpy as np


def create_lognormal_partition_sizes(dataset_name, num_partitions, sigma, seed=42):
    builder = load_dataset_builder(dataset_name)
    info = builder.info
    num_samples = info.splits["train"].num_examples
    lognormal_mean = np.log(num_samples / num_partitions)
    rnd_gen = np.random.default_rng(seed=seed)
    partition_sizes = rnd_gen.lognormal(mean=lognormal_mean, sigma=sigma,
                                        size=num_partitions)
    renomalized_partition_sizes = (partition_sizes.astype(
        int) * num_samples / partition_sizes.sum()).astype(int)
    renomalized_partition_sizes.sum()
    unassigned = num_samples - renomalized_partition_sizes.sum()
    for i in range(unassigned):
        renomalized_partition_sizes[i % num_partitions] += 1
    return renomalized_partition_sizes
