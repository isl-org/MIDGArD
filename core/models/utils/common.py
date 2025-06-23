import torch


class BatchNormLastDim(torch.nn.Module):
    """
    This module extends PyTorch's batch normalization to apply it to the last dimension of an input tensor.
    From https://github.com/JiahuiLei/NAP/blob/main/core/lib/lib_arti/articualted_denoiser_v6_2.py

    Traditional `BatchNorm1d` from PyTorch normalizes the 'feature dimension' of a 2D tensor
    (shape: `[batch, features]`) or a 3D tensor in the form `[batch, features, length]`
    by computing and subtracting the mean and dividing by the standard deviation for
    each feature independently.

    However, in certain cases, the features of interest might reside in the last dimension
    of a tensor, especially in applications dealing with sequences or timeseries data where
    the input is often of the shape `[batch, sequence, features]`.

    `BatchNormLastDim` addresses this by internally transposing the last dimension of the input tensor
    to be the second dimension, applying `BatchNorm1d`, and then transposing back to the original shape,
    thus performing batch normalization across the features located at the last dimension of the input tensor.

    Parameters:
    dim (int): The number of features expected in the input tensor's last dimension.

    Usage:
    The module can be used as a direct replacement for `torch.nn.BatchNorm1d` when the batch normalization
    needs to be applied to the last dimension of a tensor.

    Example:
        # Assuming input tensor x of shape [batch, sequence, features]
        batch_norm = BatchNormLastDim(features)
        normalized_x = batch_norm(x)

    Note:
    The input tensor `x` is expected to have more than one dimension, and batch normalization
    is applied over the last dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor):
        """
        Transpose the tensor to bring the last dimension to second dimension (feature dimension).
        Apply batch normalization.
        Transpose back to the original tensor shape.
        """
        return self.batch_norm(x.transpose(1, -1)).transpose(1, -1)


class MLP(torch.nn.Module):
    """
    A Multi-Layer Perceptron (MLP) module.
    From https://github.com/JiahuiLei/NAP/blob/main/core/lib/lib_arti/articualted_denoiser_v6_2.py

    Attributes:
        layers (torch.nn.ModuleList): A list of sequential layers making up the hidden layers of the MLP.
        output_layer (torch.nn.Linear): The final linear layer that outputs the transformed features.

    Args:
        in_dim (int): The number of features in the input data.
        out_dim (int): The number of features in the output data, i.e., the size of the output layer.
        hidden_dims (List[int]): A list specifying the number of units in each hidden layer.
        use_batch_normalization (bool): If set to True, batch normalization is applied to the output of each hidden layer.

    The MLP class is initialized by stacking several linear layers followed by LeakyReLU activation functions. If batch normalization is enabled, it is applied after each linear layer and before the activation function.

    The output layer is not followed by an activation function, making the module appropriate for regression tasks, or for use as a feature extractor within a larger model that may add its own activation function.

    Example usage:
        mlp = MLP(in_dim=128, out_dim=10, hidden_dims=[64, 64], use_batch_normalization=True)
        output = mlp(input_features)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list,
        use_batch_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        current_input_dim = in_dim

        # Create hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(current_input_dim, hidden_dim),
                    (
                        BatchNormLastDim(hidden_dim)
                        if use_batch_normalization
                        else torch.nn.Identity()
                    ),
                    torch.nn.LeakyReLU(),
                )
            )
            current_input_dim = hidden_dim

        # Concatenate all features from hidden layers and input for the output layer
        concatenated_feature_size = in_dim + sum(hidden_dims)
        self.output_layer = torch.nn.Linear(concatenated_feature_size, out_dim)

        return

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor containing features.

        Returns:
            torch.Tensor: The output tensor with transformed features after passing through the MLP.
        """
        x = x.float()
        f_list = [x]
        for l in self.layers:
            x = l(x)
            f_list.append(x)
        f = torch.cat(f_list, -1)

        return self.output_layer(f)
