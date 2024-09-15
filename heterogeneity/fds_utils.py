import pandas as pd
from datasets import DatasetDict
from flwr_datasets import FederatedDataset


def create_fds(fds_kwargs):
    if fds_kwargs["dataset"] == "flwrlabs/femnist":
        # Resplit the dataset
        fds_kwargs["preprocessor"] = resplit_femnist_to_train_test
    return FederatedDataset(**fds_kwargs)


def resplit_femnist_to_train_test(dataset):
    selected_writer_ids_for_test = pd.read_csv("./configs/test_writer_ids_femnist.csv")[
        "writer_id"
    ]
    writer_id = dataset["train"].select_columns(["writer_id"]).to_pandas()

    test_rows = writer_id["writer_id"].isin(selected_writer_ids_for_test)
    test_rows_ids = writer_id[test_rows].index
    train_rows_ids = writer_id[~test_rows].index
    dataset_test = dataset["train"].select(test_rows_ids)
    dataset_train = dataset["train"].select(train_rows_ids)
    return DatasetDict({"train": dataset_train, "test": dataset_test})
