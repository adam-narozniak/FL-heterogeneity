import itertools

import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

from heterogeneity.metrics import compute_hellinger_distance, compute_kl_divergence, compute_earths_mover_distance
# Refactor IID for different metrics
# Add Dirichlet
# Add other partitioners
#


num_partitions_to_cifar_iid_partitions = {}
num_partitions_to_cifar_iid_fds = {}
num_partitions_list = [3, 10, 30, 100, 300, 1000]
for num_partitions in num_partitions_list:
    iid_partitioner = IidPartitioner(num_partitions=num_partitions)
    cifar_iid = FederatedDataset(dataset="cifar10", partitioners={"train" : iid_partitioner})
    num_partitions_to_cifar_iid_fds[num_partitions] = cifar_iid
    # cifar_iid_partitions = [cifar_iid.load_partition(i) for i in range(num_partitions)]
    # num_partitions_to_cifar_iid_partitions[num_partitions] = cifar_iid_partitions

num_partitions_to_cifar_iid_hellinger_distance = {}
num_partitions_to_cifar_iid_hellinger_distance_list = {}
for num_partitions, cifar_iid_fds in num_partitions_to_cifar_iid_fds.items():
    metric_list, metric_avg = compute_kl_divergence(cifar_iid_fds.partitioners["train"])
    num_partitions_to_cifar_iid_hellinger_distance_list[num_partitions] = metric_list
    num_partitions_to_cifar_iid_hellinger_distance[num_partitions] = metric_avg