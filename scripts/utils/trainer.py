"""Main training loop for PyTorch models."""

# Adapted from: https://github.com/JiahuiLei/NAP/blob/main/core/solver_v2.py

import os
import gc
import numpy as np
import random
import torch
import logging
from tqdm import tqdm
from rich.console import Console

from scripts.utils.logger import Logger
from core.utils.training_utils import (
    save_experiment_params,
    load_config,
)


class Trainer:
    def __init__(
        self,
        args,
        model_cls,
        dataset_cls,
        parse_batch_fn=None,
    ) -> None:
        """
        Initialize a Trainer object.

        Args:
            args:          command line arguments.
            model_cls:     a class or function that builds the model.
            dataset_cls:   a class or function that builds the dataset.
            parse_batch_fn: optional function to pre-process or parse each batch
                            (for example, to handle 'mode', 'epoch', etc.).

        Returns:
            None
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.console = Console()
        self.console.rule("[bold]Init.")

        # Parse the config file
        self.config = load_config(args.config_file)
        self.model_cls = model_cls
        self.dataset_cls = dataset_cls
        self.parse_batch_fn = parse_batch_fn
        self.logger = Logger(self.config)

        # Extract commonly needed fields from config
        self.modes = self.config["dataset"].get("modes", ["train"])
        self.num_workers = self.config["dataset"].get("num_workers", 8)
        self.pin_memory = self.config["dataset"].get("pin_memory", True)
        self.resume_training = self.config["training"].get("resume_training", False)
        self.epochs = self.config["training"].get("total_iter", 120000)
        self.batch_size = self.config["training"].get("batch_size", 64)
        self.max_iters_per_epoch = self.config["training"].get(
            "maximum_iters_per_epoch", 1000
        )
        self.early_terminate = self.max_iters_per_epoch > 0
        self.training_drop_last = self.config["training"].get(
            "drop_last", True
        )  # drops the last non-full batch of each worker’s dataset replica.
        self.checkpoint_epoch = self.config["logging"].get("checkpoint_epoch", 1000)
        self.eval_every_iter = self.config["evaluation"].get("eval_every_iter", 1000)
        self.viz_iter_interval = self.config["logging"].get("viz_iter_interval", 1000)
        self.viz_nontrain_interval = self.config["logging"].get(
            "viz_nontrain_interval", 10
        )
        self.console = Console()

        # Check if output directory exists. If it doesn't, then create it.
        self.output_directory = os.path.join(
            self.config["base_directory"], self.config["output_directory"]
        )
        os.makedirs(self.output_directory, exist_ok=True)

        # Save the parameters of this run to a file
        save_experiment_params(
            args,
            os.path.join(
                self.output_directory,
                self.config["logging"].get("log_directory", "logs"),
            ),
        )
        logging.info("Save experiment statistics in {}".format(self.output_directory))

        # Set random seed
        self._set_random_seed(self.config["training"].get("random_seed", 42))

        # Build the dataset(s)
        logging.info(f"Preparing Dataset and Dataloaders...")
        self.datasets = self._build_datasets()
        self.dataloaders = self._build_dataloaders()
        logging.info(f"Dataset and Dataloaders ready.")

        # Build the model
        logging.info(f"Creating model.")
        self.model = self.model_cls(self.config)
        logging.info(f"Model created.")

        # Try loading checkpoint or init weights
        logging.info(f"Resuming...")
        self.current_epoch, self.batch_count = self._resume_or_initialize()

        self.console.rule("[bold]Ready.")

    def train(self):
        """
        Main training loop.
        """
        self.console.rule("[bold]Training...")

        # Initialize tqdm progress bar with a total count, which is the maximum iterations.
        progress_bar = tqdm(total=self.epochs, desc="Overall Progress")

        # Main training / evaluation loop
        logging.info("Start Training...")
        while self.batch_count <= self.epochs:
            need_val_flag = "train" not in self.modes

            for mode in self.modes:
                # Possibly skip 'val' if not the right iteration
                if mode != "train" and not need_val_flag:
                    continue

                dataloader = self.dataloaders[mode]
                batch_in_epoch_count = 0
                for batch in dataloader:
                    # Early termination of the epoch if the maximum iteration count is exceeded
                    if (
                        batch_in_epoch_count > self.max_iters_per_epoch
                        and self.early_terminate
                    ):
                        break

                    # Optionally parse
                    if self.parse_batch_fn:
                        batch = self.parse_batch_fn(
                            batch, epoch=self.current_epoch, mode=mode
                        )

                    # Increment batch counters
                    batch_in_epoch_count += 1
                    self.batch_count += 1

                    # Update progress bar
                    progress_bar.update(1)

                    # Visualization
                    visualize = False
                    if mode == "train":
                        visualize = self.batch_count % self.viz_iter_interval == 0
                    else:
                        visualize = (
                            batch_in_epoch_count % self.viz_nontrain_interval == 0
                        )

                    if mode == "train":
                        # Training routine
                        batch = self.model.train_batch(batch, visualize)
                    else:
                        # Evaluation routine
                        batch = self.model.val_batch(batch, visualize)

                    # Log iteration
                    self._log_iteration(batch, mode, batch_in_epoch_count, visualize)

                    # Checkpoint saving
                    if self.batch_count % self.checkpoint_epoch == 0:
                        self.logger.model_logger.set_save_flag(True)

                    # Check for eval
                    if self.batch_count % self.eval_every_iter == 0:
                        need_val_flag = True

                    # Adjust learning rate
                    self.model.adjust_learning_rate(self.current_epoch)

                # Log batch header data
                self.logger.log_phase()

                # Garbage collector
                gc.collect()

            # Iterate epoch
            self.current_epoch += 1

        # End logging at the end of the training loop
        self.logger.end_log()
        progress_bar.close()

        self.console.rule("[bold]Done!")

    def test(self):
        """
        Main test loop.
        """
        self.console.rule("[bold]Testing...")

    def _log_iteration(self, batch, mode, batch_in_epoch_count, visualize):
        """
        Log the current iteration.

        Args:
            batch: a batch of data, as returned by the dataloader.
            mode: the current mode of training (e.g., "train" or "val").
            batch_in_epoch_count: the current batch number in the current epoch.
            visualize: a boolean flag indicating whether to visualize the batch.

        Returns:
            None
        """
        iter_log = {
            "visualize": visualize,  # or set the logic above
            "batch": self.batch_count,
            "batch_in_epoch": batch_in_epoch_count,
            "batch_total": len(self.dataloaders[mode]),
            "epoch": self.current_epoch,
            "phase": mode.lower(),
            "output_parser": self.model.output_specs,
            "save_method": self.model.save_checkpoint,
            "meta_info": batch["meta_info"],
            "data": batch,
        }
        self.logger.log_batch(iter_log)

    def _set_random_seed(self, seed) -> None:
        """
        Set random seed for reproducibility.

        Args:
            seed (int): the seed to set.

        Returns:
            None
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = True

    def _build_datasets(self):
        """
        Build the dataset(s) and return them in a dictionary.

        Args:
            None

        Returns:
            dict: a dictionary of datasets
        """
        datasets = {}
        logging.info(f"Loading dataset.")
        for mode in self.modes:
            datasets[mode] = self.dataset_cls(self.config, mode=mode)
        logging.info(f"Dataset loaded.")

        if self.config.get("use_categorical_asset", False):
            self.config["asset_categories"] = datasets[mode].embedding_model_cats
        if self.config.get("use_categorical_body", False):
            self.config["body_categories"] = datasets[mode].embedding_body_types

        return datasets

    def _build_dataloaders(self):
        """
        Build the dataloaders for each dataset and return them in a dictionary.

        Args:
            None

        Returns:
            dict: a dictionary of dataloaders
        """
        dataloaders = {}
        logging.info(f"Creating dataloaders.")
        for mode in self.modes:
            dataset = self.datasets[mode]
            shuffle = mode == "train"
            sampler = getattr(dataset, "get_sampler", lambda: None)() or None
            collate_fn = getattr(dataset, "collate_fn", None)

            dataloaders[mode] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle and sampler is None,
                # num_workers=self.config["dataset"].get("num_workers", 8),
                pin_memory=self.config["dataset"].get("pin_memory", True),
                drop_last=(
                    mode == "train" and self.config["training"].get("drop_last", True)
                ),
                collate_fn=collate_fn,
                sampler=sampler,
            )
        logging.info(f"Dataloaders created.")
        return dataloaders

    def _resume_or_initialize(self):
        """
        Resume training from a checkpoint or initialize the model from scratch.

        Args:
            None
        Returns:
            int: current_epoch
            int: batch_count
        """
        # Load the checkpoints if they exist in the experiment directory
        loading_ignore_key = self.config["logging"].get("ignore_loading_key", [])
        if self.resume_training:
            resume_key = self.config.get("resume", "last")
            checkpoint_directory = os.path.join(
                self.output_directory,
                self.config["logging"].get("log_directory", "logs"),
                "checkpoint",
            )
            checkpoint_found = os.listdir(checkpoint_directory)
            checkpoint_fn = None
            if resume_key == "last":
                for fn in checkpoint_found:
                    if fn == "last.pt":
                        checkpoint_fn = os.path.join(checkpoint_directory, "last.pt")
            elif (
                resume_key == "finetune_shape_vqvae"
            ):  # To start from SDFusion VQVAE checkpoint
                for fn in checkpoint_found:
                    if fn == "vqvae-snet-all.pth":
                        checkpoint_fn = os.path.join(
                            checkpoint_directory, "vqvae-snet-all.pth"
                        )
            elif (
                resume_key == "finetune_sdfusion"
            ):  # To start from SDFusion UNet checkpoint
                for fn in checkpoint_found:
                    if fn == "vqvae-snet-all.pth":
                        checkpoint_fn = os.path.join(
                            checkpoint_directory, "sdfusion-mm2shape.pth"
                        )
            else:
                checkpoint_fn = os.path.join(checkpoint_directory, resume_key + ".pt")

            # Load the checkpoint
            try:
                map_fn = lambda storage, loc: storage
                logging.info("Loading checkpoint {}".format(checkpoint_fn))
                checkpoint = torch.load(checkpoint_fn, map_location=map_fn)
                logging.info("Checkpoint {} Loaded".format(checkpoint_fn))
                if "epoch" in checkpoint.keys():
                    current_epoch = checkpoint["epoch"]
                else:
                    current_epoch = 0
                if "batch" in checkpoint.keys():
                    batch_count = checkpoint["batch"]
                else:
                    batch_count = 0
                self.model.model_resume(
                    checkpoint,
                    is_initialization=False,
                    loading_ignore_key=loading_ignore_key,
                    strict=len(loading_ignore_key)
                    == 0,  # if ignore during resuming, can't use strict
                )
                print(f"Resume epoch: {current_epoch}.")
                self.model.adjust_learning_rate(current_epoch)
                current_epoch += 1
            except:
                batch_count = 0
                current_epoch = 1
                logging.warning(
                    f"Checkpoint file {checkpoint_fn} load fail, restarting training from scratch..."
                )

        elif len(self.config["training"]["initialize_network_file"]) > 0:
            assert isinstance(
                self.config["training"]["initialize_network_file"], list
            ), "Initialization from file config should be a list fo file path"
            for fn in self.config["training"]["initialize_network_file"]:
                checkpoint = torch.load(fn)
                logging.info("Initialization {} loaded.".format(fn))
                self.model.model_resume(
                    checkpoint,
                    is_initialization=True,
                    network_name=self.config["training"]["initialize_network_name"],
                    loading_ignore_key=loading_ignore_key,
                    strict=len(loading_ignore_key)
                    == 0,  # if ignore during resuming, can't use strict
                )
            batch_count = 0
            current_epoch = 1

        else:
            batch_count = 0
            current_epoch = 1

        # Copy model to device memory
        self.model.to_gpus()

        return current_epoch, batch_count
