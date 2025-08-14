"""
Taken from https://github.com/JiahuiLei/NAP/tree/main/logger
"""

import logging
import numpy as np
import os
import re
import torch


from .base_logger import BaseLogger


class CheckpointLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, config) -> None:
        super().__init__(tb_logger, log_path, config)
        self.NAME = "checkpoint"
        os.makedirs(self.log_path, exist_ok=True)
        self.phase = "train"
        self.current_epoch = -1
        self.current_batch = -1
        save_interval = config["logging"]["checkpoint_epoch"]
        if isinstance(save_interval, int):
            self.save_epoch_list = [
                i for i in range(0, self.total_epoch + 1, save_interval)
            ]
        elif isinstance(save_interval, list):
            self.save_epoch_list = save_interval
        else:
            raise RuntimeError("Checkpoint saving init invalid!")
        self.save_method = None
        # use as model select
        self.model_select_metric = config["logging"]["model_select_metric"]
        self.model_select_larger = config["logging"]["model_select_larger"]
        self.model_select_best = -np.inf if self.model_select_larger else np.inf
        self.model_select_buffer = []
        self.save_now_flag = False
        self.ignore_epoch_control = False

        self.save_latest_interval = config["logging"].get("save_latest_interval", 1)

    def log_batch(self, batch) -> None:
        self.phase = batch["phase"]
        self.current_epoch = batch["epoch"]
        self.current_batch = batch["batch"]
        if self.save_method is None:
            self.save_method = batch["save_method"]
        # update model selection metric
        if self.phase.startswith("val"):  # val or vali
            if self.model_select_metric in batch["data"]:
                metric = batch["data"][self.model_select_metric]
                if isinstance(metric, torch.Tensor):
                    metric = metric.detach().cpu()
                self.model_select_buffer.append(float(metric))

    def set_save_flag(self, ignore_epoch_control=True) -> None:
        self.save_now_flag = True  # will save at the end of the epoch during log_phase
        self.ignore_epoch_control = ignore_epoch_control  # once set, the save epoch interval will be ignored, for iter controlled version

    def log_phase(self) -> None:
        batch_epoch_info = {"batch": self.current_batch, "epoch": self.current_epoch}

        save_train_flag = self.phase == "train"
        if self.ignore_epoch_control:
            save_train_flag = save_train_flag and self.save_now_flag
        else:
            save_train_flag = (
                save_train_flag and self.current_epoch in self.save_epoch_list
            )
        if save_train_flag:  # Save a training checkpoint
            # Save the checkpoint with epoch as filename
            ckpt_filename = os.path.join(self.log_path, f"{self.current_epoch}.pt")
            self.save_method(ckpt_filename, batch_epoch_info)
            self.save_now_flag = False

            # Update (or create) symbolic link 'last.pt' to point to the newest checkpoint.
            last_symlink = os.path.join(self.log_path, "last.pt")
            if os.path.islink(last_symlink) or os.path.exists(last_symlink):
                os.remove(last_symlink)
            os.symlink(ckpt_filename, last_symlink)

            # Prune older checkpoints, keeping only the last three.
            # We assume checkpoint files follow the pattern "<epoch>.pt" (e.g. "5.pt")
            checkpoint_files = [
                f for f in os.listdir(self.log_path) if re.match(r"^\d+\.pt$", f)
            ]
            if len(checkpoint_files) > 3:
                # Sort files by epoch number (extracted from filename)
                checkpoint_files.sort(
                    key=lambda f: int(re.match(r"^(\d+)\.pt$", f).group(1))
                )
                # Remove all but the three most recent
                for old_checkpoint in checkpoint_files[:-3]:
                    os.remove(os.path.join(self.log_path, old_checkpoint))

        if self.phase.startswith("val"):  # model selection
            if len(self.model_select_buffer) > 0:
                # model select
                mean_metric = np.array(self.model_select_buffer).mean()
                select = self.better(old=self.model_select_best, new=mean_metric)
                if select:
                    # if there exist a previous best model, double check it!
                    old_fn = self.find_selected()
                    if old_fn is not None:
                        old_fn = os.path.join(self.log_path, old_fn)
                        old_metric = torch.load(old_fn)["select_metric"]
                        select = self.better(old=old_metric, new=mean_metric)
                        if select:  # remove old selection
                            os.system("rm " + old_fn)
                    if select:  # if still select
                        fn = os.path.join(self.log_path, "selected.pt")
                        batch_epoch_info["select_metric"] = mean_metric
                        self.save_method(fn, batch_epoch_info)
                        logging.info("Select epoch {} model".format(self.current_epoch))
            self.model_select_buffer = []

    def better(self, old, new) -> bool:
        select = False
        if self.model_select_larger and new > old:
            select = True
        if (not self.model_select_larger) and new < old:
            select = True
        return select

    def find_selected(self):
        ckpts = os.listdir(self.log_path)
        found = None
        for ck in ckpts:
            if ck.endswith("selected.pt"):
                found = ck
                break
        return found
