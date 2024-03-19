import numpy as np
import pandas as pd


# Maybe partitioner is the better abstraction

def hellinger_distance(fds, split):
    """Calculate Hellinger distance from all the partitions in FederatedDataset."""
    dataset = fds.load_full(split)
    all_labels = dataset.features["label"].str2int(dataset.features["label"].names)

    partitions = []
    for i in range(fds._partitioners[split]._num_partitions):
        partitions.append(fds.load_partition(i))

    # Calculate global distribution
    labels_series = pd.Series(dataset['label'])
    label_counts = labels_series.value_counts()
    label_counts_with_zeros = pd.Series(index=all_labels, data=0)
    label_counts_with_zeros = label_counts_with_zeros.add(label_counts,
                                                          fill_value=0).astype(int)
    partition_size = len(dataset)
    global_distribution = label_counts_with_zeros / partition_size

    # Calculate distributions for each client
    distributions = []
    for partition in partitions:
        labels_series = pd.Series(partition['label'])
        label_counts = labels_series.value_counts()
        label_counts_with_zeros = pd.Series(index=all_labels, data=0)
        label_counts_with_zeros = label_counts_with_zeros.add(label_counts,
                                                              fill_value=0).astype(int)
        partition_size = len(partition)
        distribution = label_counts_with_zeros / partition_size
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
