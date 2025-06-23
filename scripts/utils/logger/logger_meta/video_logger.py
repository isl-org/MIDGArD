"""
Taken from https://github.com/JiahuiLei/NAP/tree/main/logger
"""

from .base_logger import BaseLogger
import os
import numpy as np
import torch
from PIL import Image
import imageio
from datetime import datetime


class VideoLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, config) -> None:
        """
        Initializes the VideoLogger with configurations and a log path.
        """
        super().__init__(tb_logger, log_path, config)
        self.NAME = "video"
        # Ensuring the log path exists.
        os.makedirs(self.log_path, exist_ok=True)
        # Config option to determine if only one item per batch is visualized
        self.viz_one = config["logging"]["viz_one_per_batch"]
        return

    def log_batch(self, batch) -> None:
        """
        Logs video data from a batch to TensorBoard and as .gif files.

        :param batch: A dictionary containing data and metadata for the batch to log.
        """
        # Only proceed if the batch includes the video key
        if not self.NAME in batch["output_parser"]:
            return

        # Extract the relevant keys for video data
        keys_list = batch["output_parser"][self.NAME]
        if len(keys_list) == 0:
            return

        # Abort if we are not visualizing this batch
        if not batch["visualize"]:
            return

        # Data needed for logging
        data = batch["data"]
        current_epoch = batch["epoch"]
        meta_info = batch["meta_info"]

        # Create an epoch-specific directory for the log files
        epoch_dir = os.path.join(self.log_path, f"epoch_{current_epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        for video_key in keys_list:  # for each key
            if video_key not in data:
                continue

            kdata = data[video_key]

            # Ensure the right shape for batch video data (batch_size, channels, frames, height, width)
            assert len(kdata.shape) == 5, "Video data must have 5 dimensions"

            # Detach from GPU and convert to numpy if it's a torch Tensor
            nbatch = kdata.shape[0]
            if isinstance(kdata, torch.Tensor):
                kdata = kdata.detach().cpu().numpy()

            # Process each video in the batch
            for batch_id, video in enumerate(kdata):
                # Convert grayscale videos (1 channel) to RGB
                if video.shape[1] == 1:
                    video = np.concatenate([video] * 3, axis=1)

                # Rescale and clip video pixel values
                video = video * (255.0 / video.max()) if video.max() < 200 else video
                video = np.clip(video, 0, 255).astype(np.uint8)

                # Add video to TensorBoard
                self.tb.add_video(
                    f"{video_key}/{batch['phase']}",
                    torch.from_numpy(video).unsqueeze(0) / 255.0,
                    current_epoch,
                )

                # Save video frames as .gif files
                frames = [Image.fromarray(frame.transpose(1, 2, 0)) for frame in video]
                gif_path = os.path.join(epoch_dir, f"{video_key}_{current_epoch}.gif")

                # Save GIF using imageio with a fixed frames per second
                imageio.mimsave(gif_path, frames, fps=10)

                # Exit loop early if only logging one visual per batch
                if self.viz_one:
                    break

    def log_phase(self) -> None:
        """
        Placeholder for phase-logging logic, should it be needed in the future.
        """
        pass
