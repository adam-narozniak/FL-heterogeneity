from heterogeneity.metrics import (
    compute_earths_mover_distance,
    compute_hellinger_distance,
    compute_kl_divergence,
)

metrics_earths_mover = {
    "object": compute_earths_mover_distance,
    "kwargs": {"label_name": "label"},
}
metrics_hellinger = {
    "object": compute_hellinger_distance,
    "kwargs": {"label_name": "label"},
}
metrics_kl = {"object": compute_kl_divergence, "kwargs": {"label_name": "label"}}

all_metrics = [metrics_hellinger, metrics_earths_mover, metrics_kl]
