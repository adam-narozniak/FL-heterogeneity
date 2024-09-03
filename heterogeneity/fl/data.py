import warnings
from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm



def create_apply_transforms(pytorch_transforms = ToTensor(), features_name="img", label_name="label"):
    """Create a function to apply transforms to the partition from FederatedDataset."""
    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[features_name] = [pytorch_transforms(img) for img in batch[features_name]]
        return batch
    return apply_transforms

# def load_partitions(fds, features_name="img", labels_name="label"):
#     """Load partition CIFAR10 data."""
#     num_partitions = fds.partitioners["train"].num_partitions
#     partitions = []

#     def apply_transforms(batch):
#         """Apply transforms to the partition from FederatedDataset."""
#         batch[features_name] = [pytorch_transforms(img) for img in batch[features_name]]
#         return batch

#     for partition_id in range(num_partitions):
#         partition = fds.load_partition(partition_id)
#         # Divide data on each node: 80% train, 20% test
#         # partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#         pytorch_transforms = Compose(
#             [ToTensor(),]
#         )

#         partition = partition.with_transform(apply_transforms)
#         trainloader = DataLoader(partition, batch_size=32, shuffle=True)
#         # testloader = DataLoader(partition_train_test["test"], batch_size=32)

#         partitions.append(trainloader)

#     return partitions

