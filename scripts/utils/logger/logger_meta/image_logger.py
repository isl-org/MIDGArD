"""
Taken from https://github.com/JiahuiLei/NAP/tree/main/logger
"""

import os
import numpy as np
import torch
import matplotlib
from matplotlib import cm
from .base_logger import BaseLogger
from datetime import datetime
from PIL import Image

from cv2 import imwrite
import cv2
from copy import deepcopy
import imageio

# Force matplotlib to not use any Xwindows backend
matplotlib.use("Agg")


class ImageLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, config) -> None:
        """
        Initializes the ImageLogger with configurations and a log path.

        :param tb_logger: Logger to use with TensorBoard.
        :param log_path: The path where the logs should be saved.
        :param config: The configuration dictionary.
        """
        super().__init__(tb_logger, log_path, config)
        self.NAME = "image"
        # Ensure the log directory exists
        os.makedirs(self.log_path, exist_ok=True)
        # Config option to determine if only one image per batch is visualized
        self.viz_one = config["logging"]["viz_one_per_batch"]

        return

    def log_batch(self, batch) -> None:
        # get data
        if not self.NAME in batch["output_parser"].keys():
            return
        keys_list = batch["output_parser"][self.NAME]
        if len(keys_list) == 0:
            return
        data = batch["data"]
        phase = batch["phase"]
        current_epoch = batch["epoch"]
        meta_info = batch["meta_info"]

        # check whether need log
        if not batch["visualize"]:
            return

        os.makedirs(
            os.path.join(self.log_path, "epoch_%d" % current_epoch), exist_ok=True
        )

        # log
        for img_key in keys_list:  # for each key
            if img_key not in data.keys():
                continue
            kdata = data[img_key]
            if isinstance(kdata, list):
                assert len(kdata[0].shape) == 4, f"Invalid shape: {kdata[0].shape}"
                nbatch = kdata[0].shape[0]
                kdata = deepcopy(kdata)
            else:
                assert len(kdata.shape) == 4, f"Invalid shape: {kdata[0].shape}"
                nbatch = kdata.shape[0]
                if isinstance(kdata, torch.Tensor):
                    kdata = [deepcopy(kdata.detach().cpu().numpy())]
                else:
                    kdata = [deepcopy(kdata)]
            # convert to ndarray
            if isinstance(kdata[0], torch.Tensor):
                for i, tensor in enumerate(kdata):
                    kdata[i] = tensor.detach().cpu().numpy()
            # for each sample in batch
            for batch_id in range(nbatch):
                # now all cases are converted to list of image
                nview = len(kdata)
                for view_id in range(nview):
                    img = kdata[view_id][batch_id]  # 3*W*H / 1*W*H
                    assert img.ndim == 3, f"Invalid number of channels: {img.ndim}"
                    # first process image
                    color_flag = False
                    if img.shape[0] == 1:
                        color_flag = True
                        cm = matplotlib.cm.get_cmap("magma")  # ("viridis")
                        img = cm(img.squeeze(0))[..., :3]
                        img = img.transpose(2, 0, 1)
                        img *= 255
                    else:
                        img *= 255.0 if img.max() < 200 else 1
                    img = np.clip(img, a_min=0, a_max=255)
                    img = img.astype(np.uint8)
                    self.tb.add_image(
                        img_key + "/" + phase,
                        (
                            img if color_flag else img[[0, 1, 2], ...]
                        ),  # img[[2, 1, 0], ...],
                        current_epoch,
                    )
                    # save to file
                    img = img.transpose(1, 2, 0)
                    if img_key.startswith("gen_"):
                        stamp = datetime.now()
                        filename = os.path.join(
                            self.log_path,
                            "epoch_%d" % current_epoch,
                            f"generate_{stamp}_bid{batch_id}_{img_key}.png",
                        )
                    else:
                        filename = os.path.join(
                            self.log_path,
                            "epoch_%d" % current_epoch,
                            meta_info["viz_id"][batch_id]
                            + "_%d_" % (view_id)
                            + img_key
                            + ".png",
                        )
                    if color_flag:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imageio.imsave(filename, img)
                    # imwrite(filename, img)
                if self.viz_one:
                    break

    def _convert_to_numpy(self, kdata):
        """
        Converts the tensor image data to NumPy, handling both individual images and batches.

        :param kdata: Image data in either tensor format or NumPy array.
        :return: NumPy array of images.
        """
        # Handle either a list of tensors or a single tensor
        if isinstance(kdata, list):
            return [
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                for x in kdata
            ]
        elif isinstance(
            kdata, torch.Tensor
        ):  # If it's a single tensor, detach and convert to CPU
            return kdata.detach().cpu().numpy()
        else:
            return kdata

    def _process_image(self, img):
        """
        Processes the image data to be in the correct format for displaying, ensuring correct color mapping for grayscale images.

        :param img: Image data as a NumPy array.
        :return: Processed image data as a NumPy array.
        """
        # Assumes the image is in CxHxW format
        assert img.ndim == 3, "Expected image with dimensions CxHxW"

        # Convert single-channel images to a colormap (assumes single-channel images are grayscale)
        if img.shape[0] == 1:
            img = cm.magma(img[0])[
                :, :, :3
            ]  # Take the first channel and apply colormap
            img = (img * 255).astype(np.uint8)  # Rescale to [0, 255]
        else:
            img = np.clip(img, a_min=0, a_max=255).astype(np.uint8)

        return img.transpose(1, 2, 0)  # Transpose to HxWxC for saving/displaying

    def _normalize_image(self, img):
        """
        Normalizes the image data for TensorBoard.

        :param img: Image data as a NumPy array.
        :return: Normalized image data.
        """
        return (img / 255).astype(np.float32)

    def _create_filename(self, img_key, img, current_epoch, batch_id, meta_info):
        """
        Creates a filename for the image based on its key, current epoch, batch ID, and meta information.

        :param img_key: The key identifying the image.
        :param img: Image data as a NumPy array.
        :param current_epoch: Current training epoch.
        :param batch_id: Batch identifier.
        :param meta_info: Meta information dictionary.
        :return: The full path for the image file.
        """
        if img_key.startswith("gen_"):
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"generate_{stamp}_bid{batch_id}_{img_key}.png"
        else:
            filename = f"{meta_info['viz_id'][batch_id]}_{img_key}.png"

        return os.path.join(self.log_path, f"epoch_{current_epoch}", filename)

    def log_phase(self):
        """
        Placeholder for phase logging logic.
        """
        pass
