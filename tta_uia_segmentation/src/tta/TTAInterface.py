from typing import Dict
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader


class TTAStateInterface(ABC):
    """
    Interface for Test-Time Adaptation (TTA) state.

    This interface defines the methods that must be implemented by any TTA state class.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the state to initial values.
        """
        pass

    @abstractmethod
    def reset_to_state(self, state: dict) -> None:
        """
        Reset the state to the given state.

        Parameters
        ----------
        state : dict
            The state to reset to.
        """
        pass

    @abstractmethod
    def add_test_score(self, iteration: int, metric_name: str, score: float) -> None:
        """
        Add a test score to the test_scores OrderedDict.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        metric_name : str
            The name of the metric.
        score : float
            The score value.
        """
        pass

    @abstractmethod
    def add_test_loss(self, iteration: int, loss_name: str, loss_value: float) -> None:
        """
        Add a test loss to the tta_losses OrderedDict.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        loss_name : str
            The name of the loss.
        loss_value : float
            The loss value.
        """
        pass

    @abstractmethod
    def check_new_best_score(self, new_score: float) -> None:
        """
        Check if the new score is the best score and update state accordingly.

        Parameters
        ----------
        new_score : float
            The new score to be checked.
        """
        pass


class TTAInterface(ABC):

    @abstractmethod
    def tta(self, x: torch.Tensor | DataLoader) -> None:
        pass

    @abstractmethod
    def predict(
        self, x: torch.Tensor | DataLoader
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], ...]:
        pass

    @abstractmethod
    def evaluate(
        self,
        x_preprocessed: torch.Tensor,
        y_gt: torch.Tensor,
        preprocessed_pix_size: tuple[float, ...],
        gt_pix_size: tuple[float, ...],
    ) -> float | Dict[str, float]:
        pass

    @abstractmethod
    def load_state(self, path: str) -> None:
        pass

    @abstractmethod
    def save_state(self, path: str) -> None:
        pass

    @abstractmethod
    def reset_state(self) -> None:
        pass

    @abstractmethod
    def load_best_state(self) -> None:
        pass

    @abstractmethod
    def _evaluation_mode(self) -> None:
        pass

    @abstractmethod
    def _tta_fit_mode(self) -> None:
        pass

    @property
    @abstractmethod
    def tta_fitted_params(self) -> list[torch.nn.Parameter]:
        pass