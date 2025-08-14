"""
Taken from https://github.com/JiahuiLei/NAP/tree/main/logger
"""

import os
import time
import logging
import shutil
import torch
import matplotlib
from .base_logger import BaseLogger
from pprint import pformat

# Force matplotlib to not use any Xwindows backend
matplotlib.use("Agg")


class MetricLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, config) -> None:
        super().__init__(tb_logger, log_path, config)
        self.NAME = "metric"
        # Ensure necessary directories are created
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(os.path.join(self.log_path, "batchwise"), exist_ok=True)

        self.phase = None
        self.epoch = -1
        self.batch = -1
        self.batch_in_epoch = -1
        # Metric container to store metrics for averaging
        self.metric_container = dict()
        # Store starting time of the phase
        self.phase_time_start = time.time()
        # Stores the last time metrics were printed
        self.print_time = time.time()
        self.gpu_summarize_flag = config["logging"].get("gpu_summarize", True)

    def log_batch(self, batch) -> None:
        """
        Logs metrics for the current batch.
        - add each metric to tensorboard
        - record each metric for epoch save
        - display in terminal, displayed metric is averaged
        """
        if self.NAME not in batch["output_parser"]:
            return

        keys_list = batch["output_parser"][self.NAME]
        if not keys_list:
            return

        data = batch["data"]
        self.phase = batch["phase"]
        self.batch = batch["batch"]
        self.batch_in_epoch = batch["batch_in_epoch"]
        self.epoch = batch["epoch"]

        print_dict = {}
        for k in keys_list:
            if k not in data:
                continue

            self.metric_container.setdefault(k, []).append(data[k])
            self.tb_log_metric(k, data[k])

            print_dict[k] = float(data[k])

        self.print_metrics(batch, print_dict)

    def tb_log_metric(self, metric_name, value) -> None:
        """
        Logs a scalar metric to TensorBoard.
        """
        self.tb.add_scalars(
            f"Metric-BatchWise/{metric_name}", {self.phase: float(value)}, self.batch
        )

    def print_metrics(self, batch, metrics) -> None:
        """
        Prints out the logged metrics for the current batch.
        """
        # Print metrics to the terminal every 2 seconds
        if time.time() - self.print_time > 2:
            batch_size = (
                self.batch_size
                if self.phase.lower() == "train"
                else self.eval_batch_size
            )
            total_batch = batch["batch_total"]
            time_spent = (time.time() - self.phase_time_start) / 60
            time_total = time_spent / (self.batch_in_epoch + 1e-6) * total_batch
            logging.info(
                f"{self.phase} | Epoch {self.epoch}/{self.total_epoch} |"
                f" Steps {self.batch_in_epoch * batch_size}/{total_batch * batch_size} |"
                f" Time {time_spent:.3f}min/{time_total:.3f}min"
            )
            logging.info(f"Metric:\n{pformat(metrics, indent=2, compact=True)}")
            logging.info("." * 80)

            if self.gpu_summarize_flag:
                self.gpu_log()

            self.print_time = time.time()

    def gpu_log(self) -> None:
        """
        Logs GPU memory usage if enabled in the configuration.
        """
        if torch.cuda.is_available():
            used, total = torch.cuda.mem_get_info()
            mem_unit_conv = 1024**3  # Convert to GB
            logging.info(
                f"# GPU {used/mem_unit_conv:.2f}GB/{total/mem_unit_conv:.2f}GB current device #"
            )

    def log_phase(self) -> None:
        """
        At the end of each phase, logs a summary of the metrics to TensorBoard.
        """
        for k, v in self.metric_container.items():
            mean_value = sum(v) / len(v)
            self.tb.add_scalars(
                f"Metric-EpochWise/{k}", {self.phase: mean_value}, self.epoch
            )
            self.tb.add_histogram(
                f"Metric-EpochWise/{self.phase}/{k}", torch.tensor(v), self.epoch
            )

        logging.debug(
            f"Finish Epoch {self.epoch} Phase {self.phase} in {(time.time() - self.phase_time_start) / 60.0:.2f}min"
        )
        print("\n" + "=" * shutil.get_terminal_size()[0])

        # Reset metrics for the next phase
        self.metric_container.clear()
        self.phase_time_start = time.time()
