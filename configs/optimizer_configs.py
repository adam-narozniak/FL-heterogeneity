from torch.optim import SGD, Adam

sgd_config = {
    "object": [SGD],
    "lr": [0.1, 0.01, 0.001],
    "momentum": [0.9],
    "weight_decay": [0.00001, 0.0001],
}
adam_config = {"object": [Adam]}

optimizer_configs = [sgd_config, adam_config]
