from pathlib import Path

import pandas as pd


def save_fl_results(
    fds_kwargs,
    partitioner_kwargs,
    optimizer_kwargs,
    results_directory_name: str | Path,
    metrics_train_list,
    metrics_eval_list,
    metrics_aggregated_train_list,
    metrics_aggregated_eval_list,
    test_res,
):
    metrics_to_save = [
        metrics_train_list,
        metrics_eval_list,
        metrics_aggregated_train_list,
        metrics_aggregated_eval_list,
        test_res,
    ]
    metrics_names = [
        "metrics_train_list",
        "metrics_eval_list",
        "metrics_aggregated_train_list",
        "metrics_aggregated_eval_list",
        "test_res",
    ]
    for metrics_name, metric_to_save in zip(metrics_names, metrics_to_save):
        results_save_path = f"{results_directory_name}/{metrics_name}.csv"

        results_save_path = Path(results_save_path)
        include_header = True
        if results_save_path.exists():
            include_header = False
        results_save_path.parent.mkdir(parents=True, exist_ok=True)

        if metrics_name == "test_res":
            pd.DataFrame(
                [
                    [
                        *partitioner_kwargs.values(),
                        fds_kwargs.get("seed", "default"),
                        *list(metric_to_save.values()),
                    ]
                ],
                columns=[
                    *list(partitioner_kwargs.keys()),
                    "fds_seed",
                    *list(metric_to_save.keys()),
                ],
            ).to_csv(
                results_save_path,
                index=False,
                mode="a",
                header=include_header,
            )
        else:
            pd.DataFrame(
                [
                    [
                        *partitioner_kwargs.values(),
                        fds_kwargs.get("seed", "default"),
                        metric_to_save,
                    ]
                ],
                columns=[
                    *list(partitioner_kwargs.keys()),
                    "fds_seed",
                    metrics_name,
                ],
            ).to_csv(
                results_save_path,
                index=False,
                mode="a",
                header=include_header,
            )
        print(f"FL {metrics_name} saved to {results_save_path}")
