from abc import ABC, abstractmethod
from typing import List, Any, Optional

import torch

from tta_uia_segmentation.src.utils.io import save_checkpoint


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
    def checkpoint_as_dict(self) -> dict:
        """
        Returns the model checkpoint as a dictionary.
        """
        pass

    def save_checkpoint(self, path: str, **kwargs) -> None:
        save_checkpoint(
            path=path,
            **self.checkpoint_as_dict(**kwargs),
        )

    def load_checkpoint(
        self,
        path: str,
        device: Optional[str | torch.device] = None,
    ) -> None:

        checkpoint = torch.load(path, map_location=device)
        self.load_checkpoint_from_dict(checkpoint, device)
    
    @abstractmethod
    def load_checkpoint_from_dict(self, checkpoint_dict: dict, device: Optional[str | torch.device] = None) -> None:
        """
        Load model checkpoint from a dictionary.

        Parameters:
        -----------
        checkpoint_dict : dict
            Dictionary containing the model checkpoint.
        """
        pass

    def has_normalizer_module(self) -> bool:
        """
        Returns whether the model has a normalizer module.
        """
        raise NotImplementedError("BaseSeg has no implementation for has_normalizer_module")

    def get_normalizer_module(self) -> torch.nn.Module:
        """
        Returns the normalizer module of the model.
        """
        raise NotImplementedError("BaseSeg has no implementation for get_normalizer_module")

    def get_normalizer_state_dict(self) -> dict[str, Any]:
        """
        Returns the state dictionary of the normalizer module.
        """
        raise NotImplementedError("BaseSeg has no implementation for get_normalizer_module")
    
    def get_all_modules_except_normalizer(self) -> dict[str, torch.nn.Module]:
        """
        Returns all modules except the normalizer module.
        """
        raise NotImplementedError("BaseSeg has no implementation for get_all_modules_except_normalizer")
    

    @property
    def trainable_params(self) -> List[torch.nn.Parameter]:
        return [param for m in self.trainable_modules.values() for param in m.parameters()]


    @property
    @abstractmethod
    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        """
        Returns the trainable parameters of the model.
        """
        pass

    def get_bn_layers(self) -> dict[str, torch.nn.Module]:
        """
        Retrieve all nested BatchNorm layers with their module tree paths preserved.
        
        Returns:
            List of tuples containing the module path and the BatchNorm layer.
        """
        bn_layers = dict()

        def traverse(module: torch.nn.Module, path: str):
            for name, sub_module in module.named_children():
                current_path = f"{path}.{name}" if path else name
                if isinstance(sub_module, torch.nn.modules.batchnorm._BatchNorm):
                    bn_layers[current_path] = sub_module
                elif isinstance(sub_module, torch.nn.Module):
                    traverse(sub_module, current_path)

        for module_name, module in self.trainable_modules.items():
            traverse(module, module_name)

        return bn_layers

    def get_bn_layers_state_dict(self) -> dict[str, Any]:
        state_dict = dict()
        for layer_name, m in self.get_bn_layers().items():
            state_dict[layer_name] = m.state_dict()
        return state_dict       
        
    def has_bn_layers(self) -> bool:
        return len(self.get_bn_layers()) > 0
    
    def get_all_modules_except_bn_layers(self) -> dict[str, torch.nn.Module]:
        bn_layers = self.get_bn_layers()
        return {name: m for name, m in self.trainable_modules.items() if name not in bn_layers}

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
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
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
            return y_mask, y_logits, None

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

    