from torchvision.transforms import Compose, Normalize, ToTensor



def create_apply_transforms(pytorch_transforms = None, features_name="image", label_name="label"):
    """Create a function to apply transforms to the partition from FederatedDataset."""
    if pytorch_transforms is None:
        pytorch_transforms = ToTensor()
    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[features_name] = [pytorch_transforms(img) for img in batch[features_name]]
        return batch
    return apply_transforms

predefined_transforms = {
    "grayscale": Compose([ToTensor(), Normalize((0.5,), (0.5,))]),
    "rgb": Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}
