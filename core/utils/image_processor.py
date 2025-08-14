from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload, Sequence, Tuple, Optional, Union

import torch
from PIL import Image
from torchvision import transforms as T


@dataclass
class ImgProcCfg:
    """
    Configuration for ImageProcessor.

    Parameters
    ----------
    target_size        : (int, int)
        Final (H, W) after resizing.
    mean, std          : sequence(float)
        Normalisation statistics (PyTorch standard).
    pad_value          : float
        Pixel value used when padding to square.
    color_jitter       : dict | None
        kwargs for `T.ColorJitter`; `None` disables it.
    affine             : dict | None
        kwargs for `T.RandomAffine`; `None` disables it.
    hflip_prob         : float | None
        Probability for random horizontal flip; `None` disables it.
    """

    target_size: Tuple[int, int] = (256, 256)
    mean: Sequence[float] = (0.5, 0.5, 0.5)
    std: Sequence[float] = (0.5, 0.5, 0.5)
    pad_value: float = 0.0

    # Augmentation sub-configs (train-time only)
    color_jitter: Optional[dict] = field(
        default_factory=lambda: {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3}
    )
    affine: Optional[dict] = field(
        default_factory=lambda: {
            "degrees": 0,
            "scale": (0.7, 1.25),
            "interpolation": T.InterpolationMode.BILINEAR,
        }
    )
    hflip_prob: Optional[float] = 0.5


class ImageProcessor:
    """Stateless functional pipeline for training / inference."""

    @overload
    def __call__(self, img: torch.Tensor, *, train: bool = False) -> torch.Tensor: ...
    @overload
    def __call__(self, img: Image.Image, *, train: bool = False) -> torch.Tensor: ...
    @overload
    def __call__(
        self, img: Union[str, Path], *, train: bool = False
    ) -> torch.Tensor: ...

    def __init__(self, cfg: ImgProcCfg):
        """
        Initialise the image processor.

        Args:
            cfg (ImgProcCfg): Configuration for image processing.
        """
        self.cfg = cfg

        # Always-on ops
        self._to_tensor = T.ToTensor()
        self._resize = T.Resize(cfg.target_size, antialias=True)
        self._normalise = T.Normalize(cfg.mean, cfg.std)

        # Augmenters (train mode only)
        self._color = (
            T.ColorJitter(**cfg.color_jitter) if cfg.color_jitter is not None else None
        )
        self._affine = T.RandomAffine(**cfg.affine) if cfg.affine is not None else None
        self._hflip = (
            T.RandomHorizontalFlip(p=cfg.hflip_prob)
            if cfg.hflip_prob is not None
            else None
        )

    def __call__(self, img, *, train: bool = False) -> torch.Tensor:
        """
        Load, square-pad, augment (if *train*), resize and normalise.

        Args:
            img   : torch.Tensor | PIL.Image.Image | str | pathlib.Path
                * Tensor: assumed to be in the 0..1 range (CxHxW).
                * PIL: will be turned into a tensor.
                * Path: image file; loaded with PIL then converted.
            train : bool, default=False
                If True, apply data-augmentation pipeline.

        Returns:
            torch.Tensor: Processed image tensor of shape (3, H, W), float32 in -1..1 range
        """
        if isinstance(img, torch.Tensor):
            img_t = img
        elif isinstance(img, Image.Image):
            img_t = self._to_tensor(img)
        elif isinstance(img, (str, Path)):
            img_t = self._load(img)          # handles Path->tensor
        else:
            raise TypeError(
                f"Unsupported type for 'img': {type(img)!r}. "
                "Expected torch.Tensor, PIL.Image.Image or path-like object."
            )

        img_t = self._pad_to_square(img_t)

        if train:
            img_t = self._augment(img_t)

        img_t = self._resize(img_t)
        img_t = self._normalise(img_t)
        return img_t

    def unnormalize(
        self,
        img_t: torch.Tensor,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        inplace: bool = False,
    ) -> torch.Tensor:
        """Invert the Transform.Normalize applied in `__call__`.

        Parameters
        ----------
        img_t     : (C,H,W) tensor in -1…1 range.
        mean/std  : override statistics; default = cfg.mean/std.
        inplace   : modify input tensor or return a copy.

        Returns
        -------
        torch.Tensor           # pixel range back to 0…1
        """
        if not inplace:
            img_t = img_t.clone()

        mean = mean or self.cfg.mean
        std = std or self.cfg.std

        for t, m, s in zip(img_t, mean, std):
            t.mul_(s).add_(m)
        return img_t  # (C,H,W) in 0...1 for display / PIL

    def _load(self, path: str | Path) -> torch.Tensor:
        """
        PIL -> tensor in [0,1].

        Args:
            path (str | Path): Path to the image file.

        Returns:
            torch.Tensor: Image tensor of shape (3, H, W), float32 in [0,1] range
        """
        return self._to_tensor(Image.open(path).convert("RGB"))

    def _pad_to_square(self, img: torch.Tensor) -> torch.Tensor:
        """
        Zero-or-constant pad so that H == W (centre-aligned).

        Args:
            img (torch.Tensor): Image tensor of shape (3, H, W).

        Returns:
            torch.Tensor: Padded image tensor of shape (3, max(H,W), max(H,W)), float32 in [0,1] range
        """
        _, h, w = img.shape
        size = max(h, w)
        pad_h1 = (size - h) // 2
        pad_h2 = size - h - pad_h1
        pad_w1 = (size - w) // 2
        pad_w2 = size - w - pad_w1
        return torch.nn.functional.pad(
            img.unsqueeze(0),
            (pad_w1, pad_w2, pad_h1, pad_h2),
            value=self.cfg.pad_value,
        ).squeeze(0)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """
        Colour jitter + geometry preserving alpha-mask alignment.

        Args:
            img (torch.Tensor): Padded image tensor of shape (3, H, W), float32 in [0,1] range.

        Returns:
            torch.Tensor: Augmented image tensor of shape (3, H, W), float32 in [0,1] range.
        """
        # foreground mask (1 ⇔ object, 0 ⇔ padded background)
        mask = (img != self.cfg.pad_value).float().max(dim=0, keepdim=True).values
        merged = torch.cat([img, mask], dim=0)

        if self._color:
            merged[:3] = self._color(merged[:3])
        if self._affine:
            merged = self._affine(merged)
        if self._hflip:
            merged = self._hflip(merged)

        # split back
        return merged[:3]
