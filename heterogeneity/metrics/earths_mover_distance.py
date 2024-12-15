from typing import List, Tuple

from heterogeneity.metrics.utils import compute_counts, compute_distributions
from flwr_datasets.partitioner import Partitioner
import numpy as np
from scipy.stats import wasserstein_distance
from datasets import Dataset
from datasets import concatenate_datasets

def compute_earths_mover_distance(
    dataset: Dataset,
    partitions: list[Dataset],
    label_name: str = "label",
) -> Tuple[List[float], float]:

    # # Calculate global distribution
    # global_distribution = compute_distributions(dataset['label'], all_labels)
    #
    # # Calculate (local) distribution for each client
    # local_distributions = []
    # for partition in partitions:
    #     distribution = compute_distributions(partition['label'], all_labels)
    #     local_distributions.append(distribution)

    partitions_earths_mover_distance = []
    # all_labels = dataset[label_name]
    global_train_dataset = concatenate_datasets(partitions)
    all_labels = global_train_dataset[label_name]
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
        partitions_earths_mover_distance, weights=list(map(len, partitions))
    )
