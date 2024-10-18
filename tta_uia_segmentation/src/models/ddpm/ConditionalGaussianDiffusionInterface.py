from abc import ABC, abstractmethod
from typing import Optional, Literal

import torch


class ConditionalGaussianDiffusionInterface(ABC, torch.nn.Module):
    """
    Interface based on https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    
    @abstractmethod
    def forward(
        self,
        img: torch.Tensor,
        cond_img: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        min_t: Optional[int] = None,
        max_t: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model 
        
        If a vector of time steps is not passed, a random time step is sampled 
         uniformly in the range [(0|min_t), (max_t, num_timesteps)].

        If a noise tensor is not passed, it is sampled from a standard normal distribution.

        Parameters
        ----------
        img : torch.Tensor
            The input image tensor.
        cond_img : torch.Tensor
            The conditional image tensor.
        t : torch.Tensor, optional
            The time step tensor. If not passed, one is uniformly sampled.
        noise : torch.Tensor, optional
            The noise tensor. If not passed, one is sampled from a standard normal distribution.
        min_t : int, optional
            The minimum time step value to sample from.
        max_t : int, optional
            The maximum time step value to sample from.
        """  
        pass  

    @abstractmethod            
    def model_predictions(
        self,
        x: torch.Tensor,
        x_cond: torch.Tensor,
        t: torch.Tensor,
        predict: Literal['noise', 'x_start', 'v', 'pred_xt_m_1']
    ) -> torch.Tensor:
        """
        Predicts one of ['noise', 'x_start', 'v', 'pred_xt_m_1'] specified
        """
        pass    
    
    @abstractmethod
    def ddpm_sample(
        self, 
        img_shape: tuple[int, int, int, int],
        x_cond: torch.Tensor,
        return_all_timesteps: bool
    ) -> torch.Tensor:
        """
        Generate a batch of samples from the model using ancestral sampling.
        """
        pass

    @abstractmethod
    def ddim_sample(
        self, 
        img_shape: tuple[int, int, int, int],
        x_cond: torch.Tensor,
        return_all_timesteps: bool
    ) -> torch.Tensor:
        """
        Generate a batch of samples from the model using DDIM sampling.
        """
        pass

    @abstractmethod
    def sample(
        self, 
        img_shape: tuple[int, int, int, int],
        x_cond: torch.Tensor,
        return_all_timesteps: bool, 
        unconditional_sampling: bool
    ) -> torch.Tensor:
        """
        Generate a batch of samples from the model using the desired sampling method of the class.
         Allow also the option to generate a sample in unconditional model.
        """
        pass

    @property
    @abstractmethod
    def num_train_timesteps(self) -> int:
        """
        Get the number of diffusion timesteps used during training.
        """
        pass

    @property
    @abstractmethod
    def num_sample_timesteps(self) -> int:
        """
        Get the number of diffusion timesteps used during training.
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """
        Get the device of the model.
        """
        pass

    @property
    @abstractmethod
    def image_size(self) -> str:
        """
        Get the original image size used during training.

        It is assumed that the images are squares.
        """
        pass

    @property
    @abstractmethod
    def num_img_channels(self) -> int:
        """
        Get the number of channels of the images on which the model was trained 
        """
        pass

    @property
    @abstractmethod
    def also_unconditional(self) -> bool:
        """
        Get if the model was trained also in unconditional mode.
        """
        pass

    @abstractmethod
    def sample_t_noise_pairs(
        self,
        num_samples: int,
        num_imgs_per_volume: Optional[int],
        num_groups_stratified_sampling: Optional[int],
        **data_loader_kwargs
   ) -> torch.utils.data.DataLoader:
        """
        Create a dataloader with num_samples t_noise pairs 
        """
        pass

    @abstractmethod
    def train_mode():
        """
        Set the model to training mode.
        """
        pass

    @abstractmethod
    def eval_mode():
        """
        Set the model to eval mode.
        """ 
        pass

    @abstractmethod
    def enable_xformers():
        """
        Enable the xformers in the model.
        """
        pass

    @abstractmethod
    def enable_gradient_checkpointing():
        """
        Enable gradient checkpointing in the model.
        """
        pass

    @abstractmethod
    def move_non_trainable_params_to(device: str, dtype: str):
        """
        Move the non trainable parameters to the given device.
        """
        pass

    @abstractmethod
    def get_modules_to_train() -> tuple:
        """
        Get the modules that need to be trained.
        """
        pass

    @abstractmethod
    def get_params_of_modules_to_train() -> tuple:
        """
        Get the parameters of the modules that need to be trained.
        """
        pass

    @abstractmethod
    def train_mode():
        """
        Set the model to training mode.
        """
        pass

    @abstractmethod
    def eval_mode():
        """
        Set the model to eval mode.
        """ 
        pass
