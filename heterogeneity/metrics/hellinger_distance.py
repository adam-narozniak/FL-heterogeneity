import numpy as np
import pandas as pd

from heterogeneity.metrics.utils import compute_distributions, compute_counts


# Maybe partitioner is the better abstraction

def compute_hellinger_distance(fds, split, label_name: str = "label"):
    """Calculate Hellinger distance from all the partitions in FederatedDataset."""
    dataset = fds.load_split(split)
    all_labels = dataset.features[label_name].str2int(dataset.features[label_name].names)

    partitions = []
    for i in range(fds._partitioners[split]._num_partitions):
        partitions.append(fds.load_partition(i))

    # Calculate global distribution
    global_distribution = compute_distributions(dataset['label'], all_labels)

    # Calculate (local) distribution for each client
    distributions = []
    for partition in partitions:
        distribution = compute_distributions(partition['label'], all_labels)
        distributions.append(distribution)

    # Calculate Hellinger Distance
    sqrt_global_distribution = np.sqrt(global_distribution)
    sqrt_distributions = [np.sqrt(distribution) for distribution in distributions]

    hellinger_distances = []
    for sqrt_distribution in sqrt_distributions:
        diff = sqrt_global_distribution - sqrt_distribution
        diff_power_2 = np.power(diff, 2)
        hellinger_distance = np.sqrt(np.sum(diff_power_2) / 2)
        hellinger_distances.append(hellinger_distance)

    return hellinger_distances, np.average(hellinger_distances,
                                           weights=list(map(len, partitions)))
