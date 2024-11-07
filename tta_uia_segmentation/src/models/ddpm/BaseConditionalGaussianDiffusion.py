from typing import Optional, Literal

import torch


class BaseConditionalGaussianDiffusion(torch.nn.Module):
    """
    Abstract base class for conditional Gaussian diffusion models.

    Based on the interface from https://github.com/lucidrains/denoising-diffusion-pytorch

    """

    def __init__(self):
        super(BaseConditionalGaussianDiffusion, self).__init__()

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
        Forward pass of the model.

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

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def model_predictions(
        self,
        x: torch.Tensor,
        x_cond: torch.Tensor,
        t: torch.Tensor,
        predict: Literal["noise", "x_start", "v", "pred_xt_m_1"],
    ) -> torch.Tensor:
        """
        Predicts one of ['noise', 'x_start', 'v', 'pred_xt_m_1'] specified.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def ddpm_sample(
        self,
        img_shape: tuple[int, int, int, int],
        x_cond: torch.Tensor,
        return_all_timesteps: bool,
    ) -> torch.Tensor:
        """
        Generate a batch of samples from the model using ancestral sampling.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def ddim_sample(
        self,
        img_shape: tuple[int, int, int, int],
        x_cond: torch.Tensor,
        return_all_timesteps: bool,
    ) -> torch.Tensor:
        """
        Generate a batch of samples from the model using DDIM sampling.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def sample(
        self,
        img_shape: tuple[int, int, int, int],
        x_cond: torch.Tensor,
        return_all_timesteps: bool,
        unconditional_sampling: bool,
    ) -> torch.Tensor:
        """
        Generate a batch of samples from the model using the desired sampling method of the class.
        Allow also the option to generate a sample in unconditional model.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @property
    def num_train_timesteps(self) -> int:
        """
        Get the number of diffusion timesteps used during training.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @property
    def num_sample_timesteps(self) -> int:
        """
        Get the number of diffusion timesteps used during sampling.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @property
    def device(self) -> str:
        """
        Get the device of the model.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @property
    def train_image_size(self) -> str:
        """
        Get the original image size used during training.

        It is assumed that the images are squares.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @property
    def num_img_channels(self) -> int:
        """
        Get the number of channels of the images on which the model was trained.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @property
    def also_unconditional(self) -> bool:
        """
        Get if the model was trained also in unconditional mode.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @property
    def img_dtype(self) -> torch.dtype:
        """
        Get the dtype of the images on which the model was trained.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @property
    def cond_img_dtype(self) -> torch.dtype:
        """
        Get the dtype of the conditional images on which the model was trained.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def sample_t_noise_pairs(
        self,
        num_samples: int,
        num_imgs_per_volume: Optional[int],
        num_groups_stratified_sampling: Optional[int],
        **data_loader_kwargs
    ) -> torch.utils.data.DataLoader:
        """
        Create a dataloader with num_samples t_noise pairs.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def train_mode(self):
        """
        Set the model to training mode.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def eval_mode(self):
        """
        Set the model to eval mode.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def enable_xformers(self):
        """
        Enable the xformers in the model.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing in the model.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def move_non_trainable_params_to(self, device: str, dtype: str):
        """
        Move the non trainable parameters to the given device.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def get_modules_to_train(self) -> tuple:
        """
        Get the modules that need to be trained.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def get_params_of_modules_to_train(self) -> tuple:
        """
        Get the parameters of the modules that need to be trained.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def set_mixed_precision_attributes(
        self, mixed_precision_type: Literal["fp16", "bf16"]
    ):
        """
        Set the mixed precision attributes of the model.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")
