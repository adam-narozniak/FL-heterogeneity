from typing import List
import numpy as np

def fedavg(weights_list: List[np.ndarray], sample_counts: List[int]) -> List[np.ndarray]:
    """
    Computes the weighted average of a list of weights based on the sample counts.

    Args:
        weights_list (list of list of numpy arrays): List of weights, where each item is the weights 
                                                     of a single model (a list of numpy arrays).
        sample_counts (list of int): List of sample counts corresponding to each set of weights.

    Returns:
        list of numpy arrays: The weighted average of the weights.
    """
    # Check that the number of weight sets matches the number of sample counts
    if len(weights_list) != len(sample_counts):
        raise ValueError("The number of weight sets must match the number of sample counts.")

    # Total number of samples
    total_samples = sum(sample_counts)

    # Initialize the weighted sum of weights
    weighted_sum = [np.zeros_like(weight, dtype=np.float64) for weight in weights_list[0]]

    # Accumulate weighted sum of each weight set
    for weights, count in zip(weights_list, sample_counts):
        for i, weight in enumerate(weights):
            weighted_sum[i] += weight * count

    # Compute the weighted average by dividing by total samples
    averaged_weights = [w_sum / total_samples for w_sum in weighted_sum]

    return averaged_weights