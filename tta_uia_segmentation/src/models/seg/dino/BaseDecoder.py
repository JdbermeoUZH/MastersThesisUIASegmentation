import torch


class BaseDecoder(torch.nn.Module):
    def __init__(self, output_size: tuple[int, ...], *args, **kwargs):
        self._output_size = output_size
        super().__init__(*args, **kwargs)
    
    @property
    def output_size(self) -> tuple[int, ...]:
        """
        Returns the output size of the decoder.
        """
        return self._output_size

    @output_size.setter
    def output_size(self, output_size: tuple[int, ...]) -> None:
        """
        Set the output size of the decoder.
        """
        self._output_size = output_size    