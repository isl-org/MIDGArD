"""
Taken from https://github.com/JiahuiLei/NAP/tree/main/logger
"""

import os
import torch
import numpy as np
from .base_logger import BaseLogger


class HistLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, config) -> None:
        super().__init__(tb_logger, log_path, config)
        """
        Initialize the histogram logger.
        :param tb_logger: A TensorBoard logger object.
        :param log_path: The path where logs should be saved.
        :param config: Configuration dictionary.
        """
        self.NAME = "hist"
        # Ensure the logging directory exists
        os.makedirs(self.log_path, exist_ok=True)
        self.phase = None
        self.epoch = -1
        self.batch = -1
        # Container for capturing histogram data
        self.metric_container = dict()

    def log_batch(self, batch) -> None:
        """
        Log histogram data for the current batch.
        :param batch: Dictionary containing batch data and metadata.
        """
        if self.NAME not in batch["output_parser"].keys():
            return
        keys_list = batch["output_parser"][self.NAME]
        if not keys_list:  # No keys to process
            return

        # Update logger state
        self.phase = batch["phase"]
        self.batch = batch["batch"]
        self.epoch = batch["epoch"]

        for k in keys_list:
            if k not in batch["data"]:
                continue

            self._aggregate_metric(k, batch["data"][k])

    def _aggregate_metric(self, key, data) -> None:
        """
        Helper method to aggregate metrics.
        :param key: Metric key to log.
        :param data: Data associated with the key.
        """
        # Ensure there's a container for the metric
        if key not in self.metric_container:
            self.metric_container[key] = []

        if isinstance(data, (torch.Tensor, np.ndarray)):
            # Squeeze tensor to ensure it's a 1D list (for histogram)
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().squeeze()
                assert data.dim() <= 1, "Supported tensor shapes are [B] or scalar"

            # Convert to list and extend the metric container
            self.metric_container[key].extend(data.tolist())
        elif isinstance(data, (int, float)):
            # Directly append scalar values
            self.metric_container[key].append(float(data))
        else:
            raise TypeError(
                f"Unsupported type for histogram data: {type(data).__name__}"
            )

    def log_phase(self) -> None:
        """
        Log the aggregated histogram at the end of a phase.
        """
        for key, values in self.metric_container.items():
            self.tb.add_histogram(
                f"Hist/{key}/{self.phase}", torch.tensor(values), self.epoch
            )
        # Reset metric container for new phase
        self.metric_container.clear()
