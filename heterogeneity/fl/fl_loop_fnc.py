import time

import numpy as np
import ray

from heterogeneity.fl.early_stopping import EarlyStopping
from heterogeneity.fl.utils import set_weights, test, train, weighted_average
from heterogeneity.fl.weights_aggregation import fedavg


def run_fl_experiment(
    comunication_rounds,
    n_clients_per_round_train,
    n_clients_per_round_eval,
    trainloaders,
    testloaders,
    centralized_dl,
    net,
    num_local_epochs: int,
    features_name: str,
    label_name: str,
    apply_early_stopping: bool = True,
    seed: int = 42,
):
    # This is the intial model (it will be update after each communication round)
    total_num_clients = len(trainloaders)

    context = ray.init()
    print(context.dashboard_url)
    num_cpus = ray.available_resources().get("CPU")
    in_flight_tasks = num_cpus

    time_start = time.time()

    metrics_train_list = []
    metrics_eval_list = []

    metrics_aggregated_train_list = []
    metrics_aggregated_eval_list = []

    np_rng = np.random.default_rng(seed=seed)
    if apply_early_stopping:
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
                net=net,
                trainloader=trainloaders[train_client],
                epochs=num_local_epochs,
                features_name=features_name,
                label_name=label_name,
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
                    "Too many tasks in flight. This behavior should not happen."
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
        print(
            f"ROUND[{comunication_round}/{comunication_rounds}]: Global model updated"
        )

        metrics_aggregated = weighted_average(metrics_list, num_samples_list)
        print(
            f"ROUND[{comunication_round}/{comunication_rounds}]: Aggregated train metrics"
        )
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
                net=net,
                testloader=testloaders[eval_client],
                features_name=features_name,
                label_name=label_name,
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
                    "Too many tasks in flight. This behavior should not happen."
                )

        eval_res_list.extend(ray.get(eval_refs))
        metrics_eval_list.append(eval_res_list)

        # Transform eval results
        eval_num_samples_list = tuple(elem[0] for elem in eval_res_list)
        eval_metrics_list = tuple(elem[1] for elem in eval_res_list)

        eval_metrics_aggregated = weighted_average(
            eval_metrics_list, eval_num_samples_list
        )
        print(
            f"ROUND[{comunication_round}/{comunication_rounds}]: Aggregated eval metrics"
        )
        print(eval_metrics_aggregated)
        metrics_aggregated_eval_list.append(eval_metrics_aggregated)

        print(f"ROUND[{comunication_round}/{comunication_rounds}]: Finshed.")

        # Check if early stopping should be triggered
        if apply_early_stopping:
            early_stopping(
                eval_metrics_aggregated["eval_loss"], net, comunication_round
            )
            if early_stopping.early_stop:
                print(
                    f"Early stopping triggered. Stopping training after {comunication_round}."
                )
                break

    time_end = time.time()
    print(f"FL training finished in {time_end - time_start} seconds")

    print("FL Training history aggregated:")
    print(metrics_aggregated_train_list)

    print("FL Evaluation history aggregated:")
    print(metrics_aggregated_eval_list)

    if apply_early_stopping:
        print(
            "Backing up the model weigths to the best communication round"
            " (no effect if early stopping was not triggered)"
        )
        early_stopping.load_best_model(net)

    # Evaluate the final model on the test set
    print("Evaluating the final model on the test set")
    test_res = ray.get(
        test.remote(
            net=net,
            testloader=centralized_dl,
            features_name=features_name,
            label_name=label_name,
        )
    )[1]
    if apply_early_stopping:
        test_res["best_communication_round"] = early_stopping.best_round
    else:
        test_res["best_communication_round"] = comunication_rounds

    final_loss, final_acc = test_res["eval_loss"], test_res["eval_acc"]
    print(f"Final accuracy: {final_acc}, final loss: {final_loss}")

    ray.shutdown()
    return (
        metrics_train_list,
        metrics_eval_list,
        metrics_aggregated_train_list,
        metrics_aggregated_eval_list,
        test_res,
    )

    # Better to return the model in the future not to have it saved inside the fl loop fnc
    # if save_final_model:
    #     print("Saving model checkpoint")
    #     # todo adjust the path to include informatinon about the dataset and and params
    #     torch.save(net.state_dict(), f"checkpoints/model_checkpoint.pt")
