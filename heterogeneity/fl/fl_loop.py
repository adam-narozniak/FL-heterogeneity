import time

from torchvision.transforms import Compose, Normalize, ToTensor
import torch
from heterogeneity.fl.data import create_apply_transforms
from heterogeneity.fl.utils import set_weights, test, train, weighted_average
from heterogeneity.fl.early_stopping import EarlyStopping
from heterogeneity.fl.model import CNNNet
from heterogeneity.fl.weights_aggregation import fedavg
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
import ray
import numpy as np
import pandas as pd
import random
random.seed(0)


seed = 42
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

save_final_model = False


# FL-specific keywords
comunication_rounds = 1
n_clients_per_round_train = 10
n_clients_per_round_eval = 10


if __name__ == "__main__":
    num_partitions = 10
    dataset_name = "cifar10"
    total_num_clients = num_partitions
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})
    num_partitions = fds.partitioners["train"].num_partitions
    partitions = [
        fds.load_partition(partition_id) for partition_id in range(num_partitions)
    ]
    # For performance reasons flatten indices
    for partition in partitions:
        partition.flatten_indices()
    
    
    train_partitions = []
    test_partitions = []
    for partition in partitions:
        split_partition = partition.train_test_split(train_size=0.8, seed=42)
        train_partition, test_partition = (
            split_partition["train"],
            split_partition["test"],
        )
        train_partitions.append(train_partition)
        test_partitions.append(test_partition)

    if dataset_name == "mnist":
        normalization = Normalize((0.5, 0.5), (0.5, 0.5))
    else:
        normalization = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    apply_transforms = create_apply_transforms(
        Compose([ToTensor(), normalization])
    )

    trainloaders = []
    testloaders = []
    for train_partition, test_partition in zip(train_partitions, test_partitions):

        train_partition = train_partition.with_transform(apply_transforms)
        test_partition = test_partition.with_transform(apply_transforms)

        trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
        testloader = DataLoader(test_partition, batch_size=32, shuffle=False)
        trainloaders.append(trainloader)
        testloaders.append(testloader)
    centralized_ds = fds.load_split("test")
    centralized_ds = centralized_ds.with_transform(apply_transforms)
    centralized_dl = DataLoader(centralized_ds, batch_size=32, shuffle=False)

    # This is the intial model (it will be update after each communication round)
    net = CNNNet(num_classes=10)
    num_epochs = 1
    in_flight_tasks = 4
    context = ray.init(num_cpus=in_flight_tasks)
    print(context.dashboard_url)

    time_start = time.time()

    metrics_train_list = []
    metrics_eval_list = []

    metrics_aggregated_train_list = []
    metrics_aggregated_eval_list = []

    np_rng = np.random.default_rng(seed=seed)

    early_stopping = EarlyStopping()
    for comunication_round in range(1, comunication_rounds + 1):

        # Federated training
        train_refs = []
        train_res_list = []

        # Sammple n_clients_per_round_train clients to train using numpy.random.choice
        # This is a simple way to simulate the selection of clients in a federated learning setting
        selected_train_clients = np_rng.choice(
            range(total_num_clients), size=n_clients_per_round_train, replace=False
        )
        print("Selected train clients: ", selected_train_clients)
        for train_client in selected_train_clients:
            print(f"Training client {train_client}")
            train_ref = train.remote(
                net=net, trainloader=trainloaders[train_client], epochs=num_epochs
            )
            train_refs.append(train_ref)
            # This ways makes the Object Store Memory very low < 2 MB while having 1000 started at the same time makes it about 240 MB
            if len(train_refs) == in_flight_tasks:
                done_id, train_refs = ray.wait(train_refs, num_returns=1)
                assert (
                    len(done_id) == 1
                ), f"The expected len(done_id) is 1, but was {len(done_id)}"
                res = ray.get(done_id[0])
                train_res_list.append(res)
                print(res[2])
            elif len(train_refs) < in_flight_tasks:
                continue
            elif len(train_refs) > in_flight_tasks:
                raise ValueError(
                    "Too many tasks in flight. I don't know how it happend."
                )

        train_res_list.extend(ray.get(train_refs))
        metrics_train_list.append([l[1:] for l in train_res_list])

        # Transform results
        weights_list = tuple(elem[0] for elem in train_res_list)
        num_samples_list = tuple(elem[1] for elem in train_res_list)
        metrics_list = tuple(elem[2] for elem in train_res_list)

        # Aggregate the results
        global_weights = fedavg(weights_list, num_samples_list)
        set_weights(net, global_weights)
        print("Global model updated")

        metrics_aggregated = weighted_average(metrics_list, num_samples_list)
        print("Aggregated metrics")
        print(metrics_aggregated)
        metrics_aggregated_train_list.append(metrics_aggregated)

        # Federated evaluation
        eval_refs = []
        eval_res_list = []

        selected_eval_clients = np_rng.choice(
            range(total_num_clients), size=n_clients_per_round_eval, replace=False
        )
        print("Selected eval clients: ", selected_eval_clients)
        for eval_client in selected_eval_clients:
            print(f"Eval client {eval_client}")
            eval_ref = test.remote(
                net=net, testloader=testloaders[eval_client]
            )
            eval_refs.append(eval_ref)
            if len(eval_refs) == in_flight_tasks:
                done_id, eval_refs = ray.wait(eval_refs, num_returns=1)
                assert (
                    len(done_id) == 1
                ), f"The expected len(done_id) is 1, but was {len(done_id)}"
                res = ray.get(done_id[0])
                eval_res_list.append(res)
                print(res[1])
            elif len(eval_refs) < in_flight_tasks:
                continue
            elif len(eval_refs) > in_flight_tasks:
                raise ValueError(
                    "Too many tasks in flight. I don't know how it happend."
                )

        eval_res_list.extend(ray.get(eval_refs))

        metrics_eval_list.append(eval_res_list)

        # Transform eval results
        eval_num_samples_list = tuple(elem[0] for elem in eval_res_list)
        eval_metrics_list = tuple(elem[1] for elem in eval_res_list)

        eval_metrics_aggregated = weighted_average(eval_metrics_list, eval_num_samples_list)
        print(f"ROUND[{comunication_round}/{comunication_rounds}]: Aggregated eval metrics")
        print(eval_metrics_aggregated)
        metrics_aggregated_eval_list.append(eval_metrics_aggregated)


        print(f"Communication round {comunication_round} finshed.")

        # Check if early stopping should be triggered
        early_stopping(eval_metrics_aggregated["eval_loss"], net, comunication_round)
        if early_stopping.early_stop:
            print(
                f"Early stopping triggered. Stopping training after {comunication_round}."
            )
            break

    time_end = time.time()
    print(f"FL training finished in {time_end - time_start} seconds")

    pd.DataFrame([[dataset_name, metrics_train_list]], columns=["dataset_name", "metrics_train_list"]).to_csv("./fl_train_metrics.csv")
    print("FL Training history saved")


    pd.DataFrame([[dataset_name, metrics_eval_list]], columns=["dataset_name", "metrics_eval_list"]).to_csv("./fl_eval_metrics.csv")
    print("FL Evaluation history saved")

    print("FL Training history aggregated:")
    print(metrics_aggregated_train_list)
    pd.DataFrame([[dataset_name, metrics_aggregated_train_list]], columns=["dataset_name", "metrics_aggregated_train_list"]).to_csv("./fl_train_aggregated_metrics.csv")

    print("FL Evaluation history aggregated:")
    print(metrics_aggregated_eval_list)
    pd.DataFrame([[dataset_name, metrics_aggregated_eval_list]], columns=["dataset_name", "metrics_aggregated_eval_list"]).to_csv("./fl_eval_aggregated_metrics.csv")


    print(
        "Backing up the model weigths to the best communication round"
        "(no effect if early stopping was not triggered)"
    )
    early_stopping.load_best_model(net)

    # Evaluate the final model on the test set
    print("Evaluating the final model on the test set")
    test_res = ray.get(test.remote(net, centralized_dl))[1]
    final_loss, final_acc = test_res["eval_loss"], test_res["eval_acc"]
    print(f"Final accuracy: {final_acc}, final loss: {final_loss}")

    pd.DataFrame([[dataset_name, final_loss, final_acc]], columns=["dataset_name", "test_loss", "test_acc"]).to_csv("./fl_centralized_test_metrics.csv")
    ray.shutdown()
    # Save test loss and accuracy
    if save_final_model:
        print("Saving model checkpoint")
        # todo adjust the path to include informatinon about the dataset and and params
        torch.save(net.state_dict(), f"checkpoints/model_checkpoint.pt")
