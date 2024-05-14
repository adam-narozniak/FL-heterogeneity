from typing import List, Tuple

from heterogeneity.metrics.utils import compute_counts, compute_distributions
from flwr_datasets.partitioner import Partitioner
import numpy as np
from scipy.stats import wasserstein_distance


def compute_earths_mover_distance(
        partitioner: Partitioner,
        label_name: str = "label",
        aggregation_type: str = None) -> Tuple[List[float], float]:
    """

    Parameters
    ----------
    partitioner
    method: str
        "cpm-with-global", "pair-wise"

    Returns
    -------

    """
    dataset = partitioner.dataset
    # all_labels = dataset.features[label_name].str2int(
    #     dataset.features[label_name].names)

    partitions = []
    for i in range(partitioner.num_partitions):
        partitions.append(partitioner.load_partition(i))

    # # Calculate global distribution
    # global_distribution = compute_distributions(dataset['label'], all_labels)
    #
    # # Calculate (local) distribution for each client
    # local_distributions = []
    # for partition in partitions:
    #     distribution = compute_distributions(partition['label'], all_labels)
    #     local_distributions.append(distribution)

    partitions_earths_mover_distance = []
    all_labels = dataset[label_name]
    use_encoder = False
    if isinstance(all_labels[0], str):
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        all_labels = label_encoder.fit_transform(all_labels)
        use_encoder = True
    for partition in partitions:
        partition_labels = partition[label_name]
        if use_encoder:
            partition_labels = label_encoder.transform(partition_labels)
        emd = wasserstein_distance(all_labels, partition_labels)
        partitions_earths_mover_distance.append(emd)

    return partitions_earths_mover_distance, np.average(
        partitions_earths_mover_distance, weights=list(map(len, partitions)))
