"""From: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py"""

import torch

from core.models.utils.vqvae.quantizer import VectorQuantizer


def _init_weights(
    net: torch.nn.Module, init_type: str = "normal", gain: float = 0.01
) -> None:
    """
    Initialize the weights of a network recursively.

    Args:
        net (nn.Module): The network to initialize.
        init_type (str, optional): Type of initialization (e.g. "normal", "xavier", etc.). Defaults to "normal".
        gain (float, optional): Gain factor for initialization. Defaults to 0.01.

    Returns:
        None
    """

    def init_func(m: torch.nn.Module) -> None:
        classname = m.__class__.__name__
        if "BatchNorm2d" in classname or "BatchNorm3d" in classname:
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(
                    f"initialization method [{init_type}] is not implemented"
                )
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # Also propagate to children
    for m in net.children():
        m.apply(init_func)


def _normalize(in_channels: int, num_groups: int = 32) -> torch.nn.GroupNorm:
    """
    Returns a GroupNorm layer for the given number of channels.

    Args:
        in_channels (int): Number of channels in the input.
        num_groups (int, optional): Desired number of groups. Defaults to 32.

    Returns:
        nn.GroupNorm: The group normalization layer.
    """
    if in_channels <= 32:
        num_groups = max(1, in_channels // 4)
    elif in_channels % num_groups != 0:
        num_groups = 30
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


def _get_conv(data_dim: int):
    """
    Returns the appropriate convolution layer class based on spatial dimension.

    Args:
        data_dim (int): Spatial dimension (2 for 2D, 3 for 3D).

    Returns:
        type: nn.Conv2d if data_dim==2, nn.Conv3d if data_dim==3.

    Raises:
        ValueError: If data_dim is not 2 or 3.
    """
    if data_dim == 2:
        return torch.nn.Conv2d
    elif data_dim == 3:
        return torch.nn.Conv3d
    else:
        raise ValueError(f"Unsupported data_dim: {data_dim}")


class VQVAE(torch.nn.Module):
    """
    Unified Vector-Quantized Variational Auto-Encoder.
    """

    def __init__(self, config: dict) -> None:
        """
        Init VQVAE object.

        Args:
            config (dict): Configuration dictionary containing model settings.

        Returns:
            None
        """
        super().__init__()

        # Determine spatial dimension (2 or 3).
        ddconfig = config["ddconfig"]
        self.data_dim = ddconfig.get("data_dim", 2)
        self.is_voxel = self.data_dim == 3
        self.embed_dim = config.get("embed_dim", 1.0)
        self.n_embed = config.get("n_embed", 1.0)
        self.codebook_weight = config.get("codebook_weight", 1.0)
        z_channels = ddconfig.get("z_channels", 1.0)

        conv = _get_conv(self.data_dim)
        self.network_dict = torch.nn.ModuleDict(
            {
                "encoder": Encoder(**ddconfig),
                "decoder": Decoder(**ddconfig),
                "quantize": VectorQuantizer(self.n_embed, self.embed_dim, beta=1.0),
                "quant_conv": conv(z_channels, self.embed_dim, kernel_size=1),
                "post_quant_conv": conv(self.embed_dim, z_channels, kernel_size=1),
            }
        )
        # Initialize weights.
        _init_weights(self.network_dict["encoder"], "normal", 0.02)
        _init_weights(self.network_dict["decoder"], "normal", 0.02)
        _init_weights(self.network_dict["quant_conv"], "normal", 0.02)
        _init_weights(self.network_dict["post_quant_conv"], "normal", 0.02)

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encodes the input and quantizes the latent representation.

        Args:
            x (torch.Tensor): Input tensor (either an image or sdf for 3D).

        Returns:
            tuple: A tuple containing:
                - quant (torch.Tensor): Quantized latent tensor.
                - emb_loss (torch.Tensor): Embedding loss.
                - info (dict): Additional information from the quantizer.
        """
        # Encode the input
        h = self.network_dict["encoder"](x)

        # Apply convolution to the encoded data
        h = self.network_dict["quant_conv"](h)

        # Quantize the convolution-applied encoded data
        return self.network_dict["quantize"](h, is_voxel=self.is_voxel)

    def encode_no_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input without quantization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded feature tensor.
        """
        # Encode the input data
        h = self.network_dict["encoder"](x)

        # Apply convolution to the encoded data
        return self.network_dict["quant_conv"](h)

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decodes the quantized latent representation.

        Args:
            quant (torch.Tensor): Quantized latent tensor.

        Returns:
            torch.Tensor: Reconstructed output.
        """
        quant = self.network_dict["post_quant_conv"](quant)
        return self.network_dict["decoder"](quant)

    def decode_no_quant(
        self, h: torch.Tensor, force_not_quantize: bool = False
    ) -> torch.Tensor:
        """
        Optionally bypasses quantization and decodes the tensor.

        Args:
            h (torch.Tensor): Input tensor.
            force_not_quantize (bool, optional): If True, skips quantization. Defaults to False.

        Returns:
            torch.Tensor: Reconstructed output.
        """
        # If quantization is not forced to be skipped, pass 'h' through the quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.network_dict["quantize"](
                h, is_voxel=self.is_voxel
            )
        else:
            # If bypassing quantization, use 'h' directly
            quant = h

        # Apply post-quantization convolution
        quant = self.network_dict["post_quant_conv"](quant)

        # Decode the quantized tensor
        return self.network_dict["decoder"](quant)

    def forward(self, input_pack: dict, visualize: bool) -> dict:
        """
        Forward pass of the VQVAE network.

        Args:
            input_pack (dict): Dictionary with input data (key "img" for 2D or "sdf" for 3D).
            visualize (bool): Flag indicating whether to output visualization information.

        Returns:
            dict: Dictionary containing:
                - "batch_loss": Total loss.
                - "loss_codebook": Codebook loss.
                - "loss_nll": Reconstruction loss.
                - "loss_image" or "loss_sdf": Reconstruction loss for the corresponding modality.
                - "z": The quantized latent tensor.
                - "visualize": The input visualization flag.
        """
        output = {"visualize": visualize}
        # phase = input_pack["phase"]
        # epoch = input_pack["epoch"]

        if self.is_voxel:
            x_in = input_pack["sdf"]
        else:
            x_in = input_pack["img"]

        # Encode
        quant, codebook_loss, _ = self.encode(x_in)

        # Decode
        reconstruction = self.decode(quant)

        # Loss
        loss = torch.mean(torch.abs(reconstruction.contiguous() - x_in.contiguous()))

        # Output
        output["batch_loss"] = loss + self.codebook_weight * codebook_loss.mean()
        output["loss_codebook"] = codebook_loss.detach().mean()
        output["loss_nll"] = loss.detach().mean()
        if self.is_voxel:
            output["loss_sdf"] = loss.detach().mean()
        else:
            output["loss_image"] = loss.detach().mean()
        output["z"] = quant.detach()
        return output


class Upsample(torch.nn.Module):
    """
    Upsamples the input by a factor of 2 using nearest-neighbor interpolation and,
    optionally, applies a convolution.
    """

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        data_dim: int,
    ) -> None:
        """
        Init upsampling layer.

        Args:
            in_channels (int): Number of input channels.
            with_conv (bool): If True, applies a convolution after interpolation.
            data_dim (int): Spatial dimension (2 for 2D, 3 for 3D).

        Returns:
            None
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            conv = _get_conv(data_dim)
            self.conv = conv(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Upsampled (and possibly convolved) tensor.
        """
        # Use torch.nn.functional.interpolate which works for both 4D and 5D tensors.
        # Issue with Apple MPS backend: The operator 'aten::upsample_nearest3d.vec' is not currently supported
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # if x.device.type == "mps":
        #     x_cpu = x.to("cpu")
        #     upsampled_cpu = torch.nn.functional.interpolate(x_cpu, scale_factor=2.0, mode="nearest")
        #     x = upsampled_cpu.to("mps")
        # else:
        #     x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(torch.nn.Module):
    """
    Downsamples the input by a factor of 2 using strided convolution or pooling.
    """

    def __init__(self, in_channels: int, with_conv: bool, data_dim: int) -> None:
        """
        Init downsampling layer.

        Args:
            in_channels (int): Number of input channels.
            with_conv (bool): If True, applies a strided convolution.
            data_dim (int): Spatial dimension (2 for 2D, 3 for 3D).

        Returns:
            None
        """
        super().__init__()
        self.with_conv = with_conv
        self.data_dim = data_dim
        if self.with_conv:
            conv = _get_conv(data_dim)
            # For asymmetric padding (if needed), we do it manually.
            self.conv = conv(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Downsampled tensor.
        """
        if self.with_conv:
            # Apply manual padding (different for 2D vs 3D)
            if self.data_dim == 3:
                pad = (0, 1, 0, 1, 0, 1)
            else:
                pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            if self.data_dim == 3:
                x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
            else:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(torch.nn.Module):
    """
    A residual block with two convolutions and an optional dropout.
    """

    def __init__(
        self,
        *,
        data_dim: int,
        in_channels: int,
        out_channels: int = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ) -> None:
        """
        Init residual block.

        Args:
            data_dim (int): Spatial dimension (2 for 2D, 3 for 3D).
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. Defaults to in_channels.
            conv_shortcut (bool, optional): Whether to use a convolution shortcut. Defaults to False.
            dropout (float): Dropout rate.
            temb_channels (int, optional): Number of channels in the time embedding. Defaults to 512.

        Returns:
            None
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.data_dim = data_dim
        conv = _get_conv(data_dim)

        self.norm1 = _normalize(in_channels)
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = _normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = conv(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            temb (torch.Tensor, optional): Optional time embedding.

        Returns:
            torch.Tensor: Output tensor after residual addition.
        """
        h = x
        h = self.norm1(h)
        h = torch.nn.functional.silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(torch.nn.functional.silu(temb))[:, :, None, None]

        h = self.norm2(h)
        h = torch.nn.functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(torch.nn.Module):
    """
    Self-attention block.
    """

    def __init__(self, data_dim: int, in_channels: int) -> None:
        """
        Init self-attention block.

        Args:
            data_dim (int): Spatial dimension (2 for 2D, 3 for 3D).
            in_channels (int): Number of input channels.
        Returns:
            None
        """
        super().__init__()
        self.data_dim = data_dim
        conv = _get_conv(data_dim)
        self.norm = _normalize(in_channels)
        self.q = conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = conv(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with self-attention applied.
        """
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Flatten spatial dimensions.
        b, c, *spatial = q.shape
        num_elements = int(torch.tensor(spatial).prod().item())
        q = q.reshape(b, c, num_elements).permute(0, 2, 1)  # (b, num_elements, c)
        k = k.reshape(b, c, num_elements)
        v = v.reshape(b, c, num_elements)

        w_ = torch.bmm(q, k)  # (b, num_elements, num_elements)
        w_ = torch.nn.functional.softmax(w_ * (int(c) ** (-0.5)), dim=2)

        w_ = w_.permute(0, 2, 1)  # (b, num_elements, num_elements)
        h_ = torch.bmm(v, w_).reshape(b, c, *spatial)

        return x + self.proj_out(h_)


class Encoder(torch.nn.Module):
    """
    Encoder network for VQVAE (supports 2D or 3D data).
    """

    def __init__(
        self,
        *,
        data_dim: int,
        in_channels: int,
        ch: int,
        out_ch: int,
        ch_mult: list,
        num_res_blocks: int,
        attn_resolutions: list,
        dropout: float,
        resolution: int,
        z_channels: int,
        resamp_with_conv: bool = True,
        double_z: bool = True,
        activation: str = "gelu",
        **kwargs,
    ) -> None:
        """
        Init Encoder object.

        Args:
            data_dim (int): Spatial dimension (2 for 2D, 3 for 3D).
            in_channels (int): Number of input channels.
            ch (int): Base number of channels.
            out_ch (int): Number of output channels.
            ch_mult (list): Multiplicative factors for channels at each resolution.
            num_res_blocks (int): Number of residual blocks per resolution.
            attn_resolutions (list): List of resolutions where attention is applied.
            dropout (float): Dropout rate.
            resamp_with_conv (bool): If True, uses convolutions for resampling.
            resolution (int): Input resolution.
            z_channels (int): Number of channels in the latent space.
            double_z (bool, optional): Whether to double the latent channels. Defaults to True.
            activation (str, optional): Activation type ("gelu", "swish", etc.). Defaults to "gelu".

        Returns:
            None
        """
        super().__init__()
        self.data_dim = data_dim
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Select nonlinearity.
        if activation == "lrelu":
            self.nonlinearity = torch.nn.LeakyReLU()
        elif activation == "swish":
            self.nonlinearity = torch.nn.SiLU()
        elif activation == "gelu":
            self.nonlinearity = torch.nn.GELU()
        else:
            self.nonlinearity = torch.nn.SiLU()

        conv = _get_conv(data_dim)

        # Initial convolution.
        self.conv_in = conv(in_channels, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = torch.nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(
                    ResnetBlock(
                        data_dim=data_dim,
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        temb_channels=0,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    print(
                        "[*] Enc has Attn at i_level, i_block: %d, %d"
                        % (i_level, i_block)
                    )
                    attn.append(AttnBlock(data_dim=data_dim, in_channels=block_in))
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(
                    in_channels=block_in,
                    with_conv=resamp_with_conv,
                    data_dim=data_dim,
                )
                curr_res //= 2
            self.down.append(down)

        # Middle layers.
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            data_dim=data_dim,
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            temb_channels=0,
        )
        self.mid.attn_1 = AttnBlock(data_dim=data_dim, in_channels=block_in)
        self.mid.block_2 = ResnetBlock(
            data_dim=data_dim,
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            temb_channels=0,
        )
        self.norm_out = _normalize(block_in)
        self.conv_out = conv(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded latent feature tensor.
        """
        # timestep embedding
        temb = None

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions - 1:
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(torch.nn.Module):
    """
    Decoder network for VQVAE (supports 2D or 3D data).
    """

    def __init__(
        self,
        *,
        data_dim: int,
        in_channels: int,
        ch: int,
        out_ch: int,
        ch_mult: list,
        num_res_blocks: int,
        attn_resolutions: list,
        dropout: float,
        resamp_with_conv: bool,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        activation: str = "gelu",
        **kwargs,
    ) -> None:
        """
        Init Decoder object.

        Args:
            data_dim (int): Spatial dimension (2 for 2D, 3 for 3D).
            in_channels (int): Number of input channels.
            ch (int): Base number of channels.
            out_ch (int): Number of output channels.
            ch_mult (list): Multiplicative factors for channels at each resolution.
            num_res_blocks (int): Number of residual blocks per resolution.
            attn_resolutions (list): List of resolutions where attention is applied.
            dropout (float): Dropout rate.
            resamp_with_conv (bool): If True, uses convolutions for resampling.
            resolution (int): Output resolution.
            z_channels (int): Number of channels in the latent space.
            give_pre_end (bool, optional): If True, returns features before final norm and activation. Defaults to False.
            activation (str, optional): Activation type ("gelu", "swish", etc.). Defaults to "gelu".

        Returns:
            None
        """
        super().__init__()
        self.data_dim = data_dim
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        if activation == "lrelu":
            self.nonlinearity = torch.nn.LeakyReLU()
        elif activation == "swish":
            self.nonlinearity = torch.nn.SiLU()
        elif activation == "gelu":
            self.nonlinearity = torch.nn.GELU()
        else:
            self.nonlinearity = torch.nn.SiLU()

        conv = _get_conv(data_dim)

        # Compute initial block channels.
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // (2 ** (self.num_resolutions - 1))
        self.conv_in = conv(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle layers.
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            data_dim=data_dim,
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            temb_channels=0,
        )
        self.mid.attn_1 = AttnBlock(data_dim=data_dim, in_channels=block_in)
        self.mid.block_2 = ResnetBlock(
            data_dim=data_dim,
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            temb_channels=0,
        )

        # Upsampling.
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(
                    ResnetBlock(
                        data_dim=data_dim,
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        temb_channels=0,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    print(
                        "[*] Dec has Attn at i_level, i_block: %d, %d"
                        % (i_level, i_block)
                    )
                    attn.append(AttnBlock(data_dim=data_dim, in_channels=block_in))
            up = torch.nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(
                    in_channels=block_in,
                    with_conv=resamp_with_conv,
                    data_dim=data_dim,
                )
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = _normalize(block_in)
        self.conv_out = conv(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Latent tensor.

        Returns:
            torch.Tensor: Reconstructed output.
        """
        # timestep embedding
        temb = None

        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h
