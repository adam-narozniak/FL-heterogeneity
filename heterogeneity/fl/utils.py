import argparse
from typing import Dict, List, Tuple, Union
import warnings
from collections import OrderedDict
from typing import List, Dict, Union

import ray
import torch

from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


@ray.remote(num_cpus=1) # I used to have memory=100 * 10 ** 6 but it seems to be incorrectly estimated by me. I don't know why, I tried to run this script but it seems to be incorrect
def train(net, trainloader, epochs, features_name="img", labels_name="label"):
    """Train the model on the training set."""
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())# .SGD(net.parameters(), lr=0.01, momentum=0.9)#
    net.train() 
    train_loss, train_acc = 0.0, 0.0
    num_examples = len(trainloader.dataset)
    for _ in range(epochs):
        epoch_correct, epoch_loss = 0,  0.0
        for batch in trainloader:
            images = batch[features_name].to(DEVICE)
            labels = batch[labels_name].to(DEVICE)
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
    return get_weights(net), len(trainloader.dataset), {"train_loss": train_loss, "train_acc": train_acc}

@ray.remote(num_cpus=1)
def test(net, testloader, features_name="img", labels_name="label"):
    """Test the model on the test set."""
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    correct, total_loss = 0, 0.0
    net.eval()
    num_examples = len(testloader.dataset)
    with torch.no_grad():
        for batch in testloader:
            images = batch[features_name].to(DEVICE)
            labels = batch[labels_name].to(DEVICE)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / num_examples
    avg_loss = total_loss / num_examples
    return len(testloader.dataset), {"eval_loss": avg_loss, "eval_acc": accuracy}



def weighted_average(metrics_list: List[Dict[str, Union[int, float]]], 
                     num_samples_list: List[int]) -> Dict[str, Union[int, float]]:
    """
    Computes the weighted average of a list of metrics based on the number of samples.

    Args:
        metrics_list (list of dicts): List of metrics dictionaries, where each dictionary contains 
                                      metric names as keys and their corresponding values (int or float).
        num_samples_list (list of int): List of sample counts corresponding to each set of metrics.

    Returns:
        dict of str to int or float: The weighted average of the metrics.
    """
    # Check that the number of metrics sets matches the number of sample counts
    if len(metrics_list) != len(num_samples_list):
        raise ValueError("The number of metrics sets must match the number of sample counts.")

    # Total number of samples
    total_samples = sum(num_samples_list)

    # Initialize the weighted sum of metrics
    weighted_sum = {key: 0.0 for key in metrics_list[0].keys()}

    # Accumulate weighted sum of each metric set
    for metrics, count in zip(metrics_list, num_samples_list):
        for key, value in metrics.items():
            weighted_sum[key] += value * count

    # Compute the weighted average by dividing by total samples
    averaged_metrics = {key: w_sum / total_samples for key, w_sum in weighted_sum.items()}

    return averaged_metrics
