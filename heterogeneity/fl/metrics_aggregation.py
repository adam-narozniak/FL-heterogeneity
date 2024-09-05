from typing import Dict, List, Union


def weighted_average(
    metrics_list: List[Dict[str, Union[int, float]]], num_samples_list: List[int]
) -> Dict[str, Union[int, float]]:
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
        raise ValueError(
            "The number of metrics sets must match the number of sample counts."
        )

    # Total number of samples
    total_samples = sum(num_samples_list)

    # Initialize the weighted sum of metrics
    weighted_sum = {key: 0.0 for key in metrics_list[0].keys()}

    # Accumulate weighted sum of each metric set
    for metrics, count in zip(metrics_list, num_samples_list):
        for key, value in metrics.items():
            weighted_sum[key] += value * count

    # Compute the weighted average by dividing by total samples
    averaged_metrics = {
        key: w_sum / total_samples for key, w_sum in weighted_sum.items()
    }

    return averaged_metrics
