from heterogeneity.metrics.utils import compute_counts, compute_distributions
from flwr_datasets.partitioner import Partitioner
import numpy as np


def compute_kl_divergence(partitioner: Partitioner, label_name: str = "label",
                          method: str = None, aggregation_type: str = None):
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
    try:
        all_labels = dataset.features[label_name].str2int(dataset.features[label_name].names)
    except AttributeError: # Happens when the column in Value instaed of Label
        all_labels = dataset.unique(label_name)
    partitions = []
    for i in range(partitioner.num_partitions):
        partitions.append(partitioner.load_partition(i))

    # Calculate global distribution
    global_distribution = compute_distributions(dataset[label_name], all_labels)

    # Calculate (local) distribution for each client
    local_distributions = []
    for partition in partitions:
        distribution = compute_distributions(partition[label_name], all_labels)
        local_distributions.append(distribution)

    partitions_kl = []
    for partition_id, local_distribution in enumerate(local_distributions):
        global_div_local = global_distribution.divide(local_distribution)
        kl = np.sum(global_distribution.multiply(np.log(global_div_local)))
        partitions_kl.append(kl)

    return partitions_kl, np.average(partitions_kl, weights=list(map(len, partitions)))
