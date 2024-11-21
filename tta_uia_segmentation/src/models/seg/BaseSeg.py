from abc import ABC, abstractmethod
from typing import List, Any

import torch


class BaseSeg(torch.nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, x: torch.Tensor | List[torch.Tensor], **preprocess_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass of the model

        Returns:
        --------
        y_mask : torch.Tensor
            Predicted segmentation mask.

        y_logits : torch.Tensor
            Predicted logits.

        intermediate_outputs : dict[str, torch.Tensor]
            Dictionary containing intermediate outputs of preprocessing.
        """
        x_preproc, intermediate_outputs = self._preprocess_x(x, **preprocess_kwargs)

        y_mask, y_logits = self._forward(x_preproc)

        return y_mask, y_logits, intermediate_outputs

    @abstractmethod
    def select_necessary_extra_inputs(
        self, extra_input_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Selects the necessary extra inputs for the model in a dictionary
         that will be needed for the forward pass.

        """
        pass

    @abstractmethod
    def _preprocess_x(
        self, x: torch.Tensor | List[torch.Tensor]
    ) -> tuple[torch.Tensor | List[torch.Tensor], dict[str, torch.Tensor]]:
        """
        Preprocess input tensor

        Returns:
        --------
        x_preproc : torch.Tensor
            Preprocessed input tensor.

        intermediate_outputs : dict[str, torch.Tensor]
            Dictionary containing intermediate outputs of preprocessing.
        """
        pass

    @abstractmethod
    def _forward(
        self,
        x_preproc: torch.Tensor | List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model assuming a preprocessed input.

        Returns:
        --------
        y_mask : torch.Tensor
            Predicted segmentation mask.

        y_logits : torch.Tensor
            Predicted logits.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.

        Parameters:
        -----------
        path : str
            Path to save the model.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.

        Parameters:
        -----------
        path : str
            Path to load the model.
        """
        pass

    @property
    @abstractmethod
    def trainable_params(self) -> List[torch.nn.Parameter]:
        """
        Returns the trainable parameters of the model.
        """
        pass

    @property
    @abstractmethod
    def trainable_modules(self) -> List[torch.nn.Module]:
        """
        Returns the trainable parameters of the model.
        """
        pass

    @torch.inference_mode()
    def predict_mask(self, x: torch.Tensor, **preprocess_kwargs) -> torch.Tensor:
        """
        Predict segmentation mask.

        Returns:
        --------
        y_mask : torch.Tensor
            Predicted segmentation mask.
        """
        self.eval_mode()

        y_mask, _, _ = self.forward(x, **preprocess_kwargs)

        return y_mask

    @torch.inference_mode()
    def predict(
        self, x: torch.Tensor, include_interm_outs: bool = False, **preprocess_kwargs
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """
        Predict segmentation mask and logits.

        Returns:
        --------
        y_mask : torch.Tensor
            Predicted segmentation mask.

        y_logits : torch.Tensor
            Predicted logits.
        """
        self.eval_mode()

        y_mask, y_logits, interm_outs = self.forward(x, **preprocess_kwargs)

        if include_interm_outs:
            return y_mask, y_logits, interm_outs
        else:
            return y_mask, y_logits

    def eval_mode(
        self,
    ) -> None:
        """
        Sets the model to evaluation mode.
        """
        self.eval()

    def train_mode(
        self,
    ) -> None:
        """
        Sets the model to training mode.
        """
        self.train()
