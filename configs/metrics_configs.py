from heterogeneity.metrics import compute_earths_mover_distance
from heterogeneity.metrics.hellinger_distance import compute_hellinger_distance
from heterogeneity.metrics import compute_kl_divergence

metrics_earths_mover = {
    "object": compute_earths_mover_distance, "kwargs": {"label_name": "label"}
}

metrics_hellinger = {
    "object": compute_hellinger_distance, "kwargs": {"label_name": "label"}
}
metrics_kl = {
    "object": compute_kl_divergence, "kwargs": {"label_name": "label"}
}
metrics_configs = [metrics_earths_mover, metrics_hellinger, metrics_kl]
