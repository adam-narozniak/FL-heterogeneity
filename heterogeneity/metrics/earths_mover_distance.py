from heterogeneity.metrics.utils import compute_counts, compute_distributions
from flwr_datasets.partitioner import Partitioner
import numpy as np
from scipy.stats import wasserstein_distance


def compute_earths_mover_distance(partitioner: Partitioner, label_name: str=None, aggregation_type: str = None):
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
    for partition in partitions:
        emd = wasserstein_distance(dataset["label"], partition["label"])
        partitions_earths_mover_distance.append(emd)

    return partitions_earths_mover_distance, np.average(partitions_earths_mover_distance, weights=list(map(len, partitions)))
