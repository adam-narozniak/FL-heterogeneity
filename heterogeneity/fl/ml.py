import ray
import torch

from heterogeneity.fl.ml_utils import DEVICE, get_weights


@ray.remote(num_cpus=1)
def train(net, trainloader, epochs, optimizer_class, optimizer_kwargs, features_name="img", label_name="label"):
    """Train the model on the training set."""
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = optimizer_class(net.parameters(), **optimizer_kwargs)
    net.train()
    train_loss, train_acc = 0.0, 0.0
    num_examples = len(trainloader.dataset)
    for _ in range(epochs):
        epoch_correct, epoch_loss = 0, 0.0
        for batch in trainloader:
            images = batch[features_name].to(DEVICE)
            labels = batch[label_name].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= num_examples
        epoch_acc = epoch_correct / num_examples
        train_loss += epoch_loss
        train_acc += epoch_acc
    train_loss /= epochs
    train_acc /= epochs
    return (
        get_weights(net),
        len(trainloader.dataset),
        {"train/loss": train_loss, "train/acc": train_acc},
    )


@ray.remote(num_cpus=1)
def test(net, testloader, features_name="img", label_name="label"):
    """Test the model on the test set."""
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    correct, total_loss = 0, 0.0
    net.eval()
    num_examples = len(testloader.dataset)
    with torch.no_grad():
        for batch in testloader:
            images = batch[features_name].to(DEVICE)
            labels = batch[label_name].to(DEVICE)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / num_examples
    avg_loss = total_loss / num_examples
    return len(testloader.dataset), {"eval/loss": avg_loss, "eval/acc": accuracy}
