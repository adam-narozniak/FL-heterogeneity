from typing import Dict, Optional

import torch
from torch import nn


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: bool = False,
        mode: str = "min",
    ) -> None:
        """
        Early stopping to stop training when a monitored metric stops improving.

        Args:
            patience (int): How long to wait after the last improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message when early stopping is triggered.
            mode (str): 'min' for monitoring loss (lower is better), 'max' for accuracy (higher is better).
        """
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.best_round: Optional[int] = None
        self.early_stop: bool = False
        self.mode: str = mode
        self.sign: int = (
            -1 if mode == "min" else 1
        )  # Minimize loss or maximize accuracy.
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = (
            None  # In-memory storage of the best model state.
        )

    def __call__(self, score: float, model: nn.Module, best_round: int) -> None:
        """
        Call method to update the early stopping logic.

        Args:
            score (float): The current score to evaluate for improvement.
            model (nn.Module): The model whose weights to save if improvement is detected.
        """
        current_score: float = self.sign * score

        if self.best_score is None:
            self.best_score = current_score
            self.best_round = best_round
            self._save_best_model_state(model)
        elif (
            current_score < self.best_score + self.min_delta
        ):  # Adjust direction based on sign.
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.best_round = best_round
            self._save_best_model_state(model)
            self.counter = 0

    def _save_best_model_state(self, model: nn.Module) -> None:
        """
        Save the current state of the model as the best state so far in memory.

        Args:
            model (nn.Module): The model whose weights are saved in memory.
        """
        self.best_model_state = {
            k: v.clone() for k, v in model.state_dict().items()
        }  # Save a deep copy of the weights
        if self.verbose:
            print("Model improved. Best model state updated in memory.")

    def load_best_model(self, model: nn.Module) -> None:
        """
        Load the best model state saved in memory during early stopping.

        Args:
            model (nn.Module): The model to load the saved state into.
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print("Best model weights loaded.")
        else:
            print("No best model state found to load.")
