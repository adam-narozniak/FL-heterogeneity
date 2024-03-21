import pandas as pd


def compute_counts(labels, all_labels):
    labels_series = pd.Series(labels)
    label_counts = labels_series.value_counts()
    label_counts_with_zeros = pd.Series(index=all_labels, data=0)
    label_counts_with_zeros = label_counts_with_zeros.add(label_counts,
                                                          fill_value=0).astype(int)
    return label_counts_with_zeros


def compute_distributions(labels, all_labels):
    counts = compute_counts(labels, all_labels)
    counts = counts.div(len(labels))
    return counts
