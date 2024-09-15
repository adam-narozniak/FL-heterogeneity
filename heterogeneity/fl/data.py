from typing import List, Tuple

from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

predefined_transforms = {
    "grayscale": Compose([ToTensor(), Normalize((0.5,), (0.5,))]),
    "rgb": Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}


def create_apply_transforms(
    pytorch_transforms=None, features_name="image", label_name="label"
):
    """Create a function to apply transforms to the partition from FederatedDataset."""
    if pytorch_transforms is None:
        pytorch_transforms = ToTensor()

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[features_name] = [pytorch_transforms(img) for img in batch[features_name]]
        return batch

    return apply_transforms


def create_dataloaders(
    fds: FederatedDataset, features_name: str, label_name: str, seed: int = 42
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    num_partitions = fds.partitioners["train"].num_partitions
    partitions = [
        fds.load_partition(partition_id) for partition_id in range(num_partitions)
    ]

    # For performance reasons flatten indices (it would work better if they were not shuffled during each train)
    # for partition_id, partition in enumerate(partitions):
    #     partitions[partition_id] = partition.flatten_indices()

    image_mode = (
        "grayscale" if fds.load_split("train").info.dataset_name in ["mnist", "femnist"] else "rgb"
    )

    apply_transforms = create_apply_transforms(
        predefined_transforms[image_mode],
        features_name=features_name,
        label_name=label_name,
    )

    centralized_ds = fds.load_split("test")
    centralized_ds = centralized_ds.with_transform(apply_transforms)
    train_partitions = []
    test_partitions = []
    for partition in partitions:
        split_partition = partition.train_test_split(train_size=0.8, seed=seed)
        train_partition, test_partition = (
            split_partition["train"],
            split_partition["test"],
        )
        train_partitions.append(train_partition)
        test_partitions.append(test_partition)

    trainloaders = []
    testloaders = []
    for train_partition, test_partition in zip(train_partitions, test_partitions):
        train_partition = train_partition.with_transform(apply_transforms)
        test_partition = test_partition.with_transform(apply_transforms)

        trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
        testloader = DataLoader(test_partition, batch_size=32, shuffle=False)
        trainloaders.append(trainloader)
        testloaders.append(testloader)

    centralized_dl = DataLoader(centralized_ds, batch_size=32, shuffle=False)

    return trainloaders, testloaders, centralized_dl
