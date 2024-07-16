import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAdapter(nn.Module):
    """
    Args:
        input_dim (int): Dimension of input features
        adapter_dim (int): Intermediate dimension for adapter
        use_gelu (bool): Flag for using gelu activation or not. Default is False
    """

    def __init__(self, input_dim: int, adapter_dim: int, use_gelu=False):
        super().__init__()

        self.input_dim = input_dim
        self.adapter_dim = adapter_dim

        self.fc1 = nn.Linear(self.input_dim, self.adapter_dim)
        self.fc2 = nn.Linear(self.adapter_dim, self.input_dim)
        self.activation = nn.GELU() if use_gelu else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pretrained Features
        Returns: Adapted Features for the task
        """
        orig_shape = x.shape
        
        # Flatten the spatial_dimension such that x -> [B*H*W*D, C], if x is 5D tensors i.e. x -> [B, H, W, D, C]
        if len(orig_shape) == 5:
            x_flat = x.view(-1, self.input_dim)
        else:
            raise ValueError(f"Unexpected input shape: {orig_shape}")
        
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))

        # Reshape back to original dimension
        if len(orig_shape) == 5:
            h = h.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], self.input_dim)

        return x + h