import copy
import logging
import os
import platform
import torch

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class NeuralModel:
    """
    A wrapper class for neural network models containing a bunch of useful routines.

    Args:
        config (dict): Configuration dictionary with required settings.
        network (torch.nn.Module): The neural network module to be wrapped.

    Raises:
        ValueError: If config is missing required fields.
    """

    def __init__(self, config: dict, neural_network: torch.nn.Module) -> None:
        # Initialization
        self.config = copy.deepcopy(config)  # Configuration parameters
        self.__dataparallel_flag__ = False  # Account for multi-gpu clusters
        self.neural_network = neural_network  # Dictionary of neural networks
        os_type = platform.system()
        if os_type == "Darwin":
            self.device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        elif os_type == "Linux" or os_type == "Windows":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.xpu.is_available():
                self.device = torch.device("xpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if "training" in self.config:
            # Setup optimizer configurationGraphTransformerLayermidgard
            self.optimizer_specs = self.config["training"]["optimizer"]
            # Each element of neural_network.network_dict potentially has its own optimizer/scheduler
            self.optimizer_dict, self.lr_scheduler_dict = self._setup_optimizer()
            # If True, automatically calls 'adjust_learning_rate' at the end of a training epoch
            self.auto_lr_adjust = self.config["training"].get("auto_lr_adjust", False)

            # Gradient and loss clips
            self.grad_clip = float(self.config["training"].get("grad_clip", 1e9))
            self.loss_clip = float(self.config["training"].get("loss_clip", 1e9))
        else:
            self.optimizer_dict, self.lr_scheduler_dict = {}, {}

        self.train_epoch = 0
        self.val_epoch = 0
        self.epoch = 0

        # Data container for logging
        self.output_specs = {
            "metric": [],
        }

        # Log the number of parameters
        self._count_parameters()

        return

    def _count_parameters(self) -> None:
        """
        Calculate and log the number of parameters and the number of trainable parameters
        for each part of the neural network.

        This method accounts for whether the neural network is wrapped with `torch.nn.DataParallel`
        or not, to access the appropriate network dictionary. It iterates through each component of the
        network, calculates the total number of parameters and the number of trainable parameters,
        and logs this information. This is useful for understanding the size and complexity of the model,
        as well as the proportion of parameters that can be trained (learned) during the training process.
        """
        # Determine the correct network dictionary to use based on the DataParallel flag
        neural_network_dict = (
            self.neural_network.module.network_dict
            if self.__dataparallel_flag__
            else self.neural_network.network_dict
        )

        # Iterate through each component of the network
        for k, v in neural_network_dict.items():
            # Count the total number of parameters in the component
            num_params = sum(p.numel() for p in v.parameters())
            # Count the number of trainable parameters in the component
            num_trainable_params = sum(
                p.numel() for p in v.parameters() if p.requires_grad
            )
            # Log the count of parameters and trainable parameters
            logging.info(
                "The model {} has {:.3f} millions of parameters, from which {:.3f} millions are trainable.".format(
                    k, num_params / 1e6, num_trainable_params / 1e6
                )
            )

        return

    def _setup_optimizer(self) -> dict:
        """
        Generate a suitable torch optimizer configuration based on the provided config dictionary.

        Check https://pytorch.org/docs/stable/optim.html#module-torch.optim
        """
        optimizer_dict = {}
        lr_scheduler_dict = {}
        optimizer_config_keys = self.optimizer_specs.keys()
        logging.debug(
            "Config defines {} network parameters optimization".format(
                optimizer_config_keys
            )
        )

        # If the number of optimizers does not match the number of neural modules in self.neural_network.network_dict
        if len(optimizer_config_keys) != len(self.neural_network.network_dict.keys()):
            logging.warning("Network Components != Optimizer Config")

        try:
            # If all optimizers share the same configutation
            if (
                "all" in optimizer_config_keys
            ):  # The 'all' configuration preempts all other configurations
                optimizer_config_keys = ["all"]

            # Loop throught the different optimizer configs
            for optimizer_config_key in optimizer_config_keys:
                optimizer_params = self.optimizer_specs[optimizer_config_key]
                optimizer_type = optimizer_params.get("type", "adam").lower()
                if optimizer_type == "sgd":
                    # Implements stochastic gradient descent (optionally with momentum).
                    optimizer = torch.optim.SGD(
                        self.neural_network.parameters(),  # iterable of parameters to optimize or dicts defining parameter groups
                        lr=optimizer_params.get("lr", 1e-3),  # learning rate
                        weight_decay=optimizer_params.get(
                            "weight_decay", 0.0
                        ),  # weight decay (L2 penalty)
                        momentum=optimizer_params.get(
                            "momentum", 0.0
                        ),  # momentum factor
                        dampening=optimizer_params.get(
                            "dampening", 0.0
                        ),  # dampening for momentum
                        nesterov=optimizer_params.get("nesterov", False),
                    )  # enables Nesterov momentum
                elif optimizer_type == "adam":
                    # Implements Adam algorithm.
                    optimizer = torch.optim.Adam(
                        self.neural_network.parameters(),  # iterable of parameters to optimize or dicts defining parameter groups
                        lr=optimizer_params.get("lr", 1e-3),  # learning rate
                        weight_decay=optimizer_params.get(
                            "weight_decay", 0.0
                        ),  # weight decay (L2 penalty)
                        betas=optimizer_params.get("betas", (0.9, 0.999)),
                        amsgrad=optimizer_params.get(
                            "amsgrad", False
                        ),  # whether to use the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond"
                        maximize=optimizer_params.get("maximize", False),
                    )  # maximize the params based on the objective, instead of minimizing
                elif optimizer_type == "adamw":
                    # Implements Adam algorithm.
                    optimizer = torch.optim.AdamW(
                        self.neural_network.parameters(),  # iterable of parameters to optimize or dicts defining parameter groups
                        lr=optimizer_params.get("lr", 1e-3),  # learning rate
                        eps=optimizer_params.get(
                            "eps", 1e-8
                        ),  # Term added to the denominator to improve numerical stability
                        weight_decay=optimizer_params.get(
                            "weight_decay", 0.0
                        ),  # weight decay (L2 penalty)
                        betas=optimizer_params.get("betas", (0.9, 0.999)),
                        amsgrad=optimizer_params.get(
                            "amsgrad", False
                        ),  # whether to use the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond"
                        maximize=optimizer_params.get("maximize", False),
                    )  # maximize the params based on the objective, instead of minimizing
                elif optimizer_type == "radam":
                    # Implements RAdam algorithm.
                    optimizer = torch.optim.RAdam(
                        self.neural_network.parameters(),  # iterable of parameters to optimize or dicts defining parameter groups
                        lr=optimizer_params.get("lr", 1e-3),  # learning rate
                        weight_decay=optimizer_params.get("weight_decay", 0.0),
                    )  # weight decay (L2 penalty)
                elif optimizer_type == "nadam":
                    # Implements NAdam algorithm.
                    optimizer = torch.optim.RAdam(
                        self.neural_network.parameters(),  # iterable of parameters to optimize or dicts defining parameter groups
                        lr=optimizer_params.get("lr", 1e-3),  # learning rate
                        weight_decay=optimizer_params.get(
                            "weight_decay", 0.0
                        ),  # weight decay (L2 penalty)
                        momentum_decay=optimizer_params.get("momentum_decay", 4e-3),
                    )  # momentum momentum_decay
                else:
                    raise NotImplementedError()

                optimizer_dict[optimizer_config_key] = optimizer
                lr_scheduler_dict[optimizer_config_key] = self._setup_lr_scheduler(
                    optimizer, optimizer_params["scheduler"]
                )

        except Exception as e:
            logging.error(f"Error in the neural model optimizer setup: {e}")
            raise

        return optimizer_dict, lr_scheduler_dict

    def _create_lambda_function(self, config: dict):
        """
        Create a lambda scheduling function for the lambda schedulers
        """
        decay_factor = config["decay_factor"]
        decay_schedule = config["decay_schedule"]
        lr_min = config["lr_min"]
        self.decay_factor = 1.0

        def lr_lambda(epoch) -> float:
            if epoch in decay_schedule:
                setattr(self, "decay_factor", decay_factor[decay_schedule.index(epoch)])

            return self.decay_factor

        return lr_lambda

    def _create_multiplicative_function(self, config: dict):
        """
        Create a lambda scheduling function for the multiplicative schedulers
        """
        start_epoch = config.get("start_epoch", 0)
        end_epoch = config.get("end_epoch", 10)
        start_factor = config.get("start_factor", 1.0)
        end_factor = config.get("end_factor", 0.5)

        def lr_lambda(epoch):
            if epoch < start_epoch:
                return start_factor
            elif start_epoch <= epoch <= end_epoch:
                return start_factor + (end_factor - start_factor) * (
                    (epoch - start_epoch) / (end_epoch - start_epoch)
                )
            else:
                return end_factor

        return lr_lambda

    def _setup_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: dict,
    ) -> torch.optim.lr_scheduler:
        """
        Generate a suitable torch optimizer configuration based on the provided config dictionary.

        Check https://pytorch.org/docs/stable/optim.html#module-torch.optim
        """
        try:
            schedule_type = config.get("schedule_type", "constant").lower()
            if schedule_type == "constant":
                # Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches a pre-defined milestone: total_iters.
                # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html#torch.optim.lr_scheduler.ConstantLR
                scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer,  # Wrapped optimizer.
                    factor=config["constant"].get(
                        "factor", 0.3333333333333333
                    ),  # The number we multiply learning rate until the milestone.
                    total_iters=config["constant"].get("total_iters", 5),
                    last_epoch=config["constant"].get("last_epoch", -1),
                )  # If 'True', prints a message to stdout for each update.
            elif schedule_type == "step":
                # Decays the learning rate of each parameter group by 'gamma' every 'step_size' epochs.
                # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,  # Wrapped optimizer.
                    step_size=config["step"].get("step_size", 100),
                    gamma=config["step"].get(
                        "gamma", 0.1
                    ),  # Multiplicative factor of learning rate decay.
                    last_epoch=config["step"].get("last_epoch", -1),
                )  # If 'True', prints a message to stdout for each update.
            elif schedule_type == "multistep":
                # Decays the learning rate of each parameter group by 'gamma' once the number of epoch reaches one of the 'milestones'.
                # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,  # Wrapped optimizer.
                    milestones=config["multistep"].get("milestones", []),
                    gamma=config["multistep"].get(
                        "gamma", 0.1
                    ),  # Multiplicative factor of learning rate decay.
                    last_epoch=config["multistep"].get("last_epoch", -1),
                )  # If 'True', prints a message to stdout for each update.
            elif schedule_type == "multiplicative":
                # Multiply the learning rate of each parameter group by the factor given in the specified function.
                # The lambda function takes the current learning rate as an input and returns the new learning rate factor.
                # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR
                lambda_params = config["lambda"].get("lambda_params", {})
                lambda_func = self._create_multiplicative_function(
                    lambda_params
                )  # A function which computes a multiplicative factor given an integer parameter
                scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                    optimizer,  # Wrapped optimizer.
                    lr_lambda=lambda_func,
                    last_epoch=config["multiplicative"].get("last_epoch", -1),
                )  # If 'True', prints a message to stdout for each update.
            elif schedule_type == "lambda":
                # Sets the learning rate of each parameter group to the initial lr times a given function.
                # The lambda function takes an integer (epoch) and returns a multiplicative factor.
                # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR
                lambda_params = config["lambda"].get("lambda_params", {})
                lambda_func = self._create_lambda_function(
                    lambda_params
                )  # A function which computes a multiplicative factor given an integer parameter
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,  # Wrapped optimizer.
                    lr_lambda=lambda_func,
                    last_epoch=config["lambda"].get("last_epoch", -1),
                )  # If 'True', prints a message to stdout for each update.
            elif schedule_type == "linear":
                # Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
                # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,  # Wrapped optimizer.
                    start_factor=config["linear"].get(
                        "start_factor", 0.3333333333333333
                    ),  # The number we multiply learning rate in the first epoch. The multiplication factor changes towards 'end_factor' in the following epochs.
                    end_factor=config["linear"].get(
                        "end_factor", 1.0
                    ),  # The number we multiply learning rate at the end of linear changing process.
                    total_iters=config["linear"].get(
                        "toral_iters", 5
                    ),  # The number of iterations that multiplicative factor reaches to 1.
                    last_epoch=config["linear"].get(
                        "last_epoch", -1
                    ),  # The index of the last epoch.
                )  # If 'True', prints a message to stdout for each update.
            elif schedule_type == "exponential":
                # Decays the learning rate of each parameter group by gamma every epoch.
                # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,  # Wrapped optimizer.
                    gamma=config["exponential"].get(
                        "gamma", 0.1
                    ),  # Multiplicative factor of learning rate decay.
                    last_epoch=config["exponential"].get(
                        "last_epoch", -1
                    ),  # The index of the last epoch.
                )  # If 'True', prints a message to stdout for each update.

            elif schedule_type == "polynomial":
                # Decays the learning rate of each parameter group using a polynomial function in the given total_iters.
                # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR
                scheduler = torch.optim.lr_scheduler.PolynomialLR(
                    optimizer,  # Wrapped optimizer.
                    total_iters=config["polynomial"].get(
                        "total_iters", 1
                    ),  # The number of steps that the scheduler decays the learning rate.
                    power=config["polynomial"].get(
                        "power", 1.0
                    ),  # The power of the polynomial.
                    last_epoch=config["polynomial"].get(
                        "last_epoch", -1
                    ),  # The index of the last epoch.
                )  # If 'True', prints a message to stdout for each update.
            elif schedule_type == "cosineanneal":
                # Set the learning rate of each parameter group using a cosine annealing schedule, where 'eta_max'​ is set to the initial lr and 'Tcur'​ is the number of epochs since the last restart in SGDR
                # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,  # Wrapped optimizer.
                    T_max=config["cosineanneal"].get(
                        "T_max", 1
                    ),  # Maximum number of iterations.
                    eta_min=config["cosineanneal"].get(
                        "eta_min", 0
                    ),  # Minimum learning rate.
                    last_epoch=config["cosineanneal"].get(
                        "last_epoch", -1
                    ),  # The index of the last epoch.
                )  # If 'True', prints a message to stdout for each update.
            elif schedule_type == "cosineannealwarmrestarts":
                # Set the learning rate of each parameter group using a cosine annealing schedule, where 'eta_max'​ is set to the initial lr, 'Tcur'​ is the number of epochs since the last restart and
                # Ti​ is the number of epochs between two warm restarts in SGDR
                # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,  # Wrapped optimizer.
                    T_0=config["cosineannealwarmrestarts"].get(
                        "T_0", 1
                    ),  # Number of iterations for the first restart.
                    T_mult=config["cosineannealwarmrestarts"].get(
                        "T_mult", 1
                    ),  # A factor increases 'Ti'​ after a restart.
                    eta_min=config["cosineannealwarmrestarts"].get(
                        "eta_min", 0
                    ),  # Minimum learning rate.
                    last_epoch=config["cosineannealwarmrestarts"].get(
                        "last_epoch", -1
                    ),  # The index of the last epoch.
                )  # If 'True', prints a message to stdout for each update.
            else:
                NotImplementedError()
        except Exception as e:
            logging.error(f"Error in the neural model lr_scheduler setup: {e}")
            raise

        return scheduler

    def adjust_learning_rate(self, epoch: int) -> None:
        """
        Adjust the learning rate using the existing scheduler.
        """
        try:
            for key in self.lr_scheduler_dict.keys():
                self.lr_scheduler_dict[key].step(epoch)
            return
        except Exception as e:
            logging.error(f"Error in the neural model lr_scheduler.step routine: {e}")
            raise

    def _preprocess(self, data_batch: tuple, visualize: bool = False) -> dict:
        """
        Preprocess the input data before feeding it into the model.

        Args:
            data_batch (tuple): The raw input data.
            visualize (bool): A flag indicating whether to include additional visualization data in the output.

        Returns:
            dict: Preprocessed data ready for model input.
        """
        try:
            data, meta_info = data_batch

            # Copy tensor data onto the GPU memory
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(self.device, dtype=torch.float32)

            # Phase ['train', 'val']
            data["phase"] = meta_info["mode"][0]
            data["visualize"] = visualize
            return {"model_input": data, "meta_info": meta_info}

        except Exception as e:
            logging.error(f"Error in the neural model preprocessing step: {e}")
            raise

    def _propagate(self, data_batch: dict, visualize: bool = False) -> dict:
        """
        Propagate the processed data through the network

        Args:
            data_batch (dict): Data processed by _preprocess.
            visualize (bool): A flag indicating whether to include additional visualization data in the output.

        Returns:
            dict: The prediction output from the model.
        """
        try:
            if self.__dataparallel_flag__:
                propagated_data = self.neural_network.module(
                    data_batch["model_input"], visualize
                )
            else:
                propagated_data = self.neural_network(
                    data_batch["model_input"], visualize
                )

            for key, val in propagated_data.items():
                data_batch[key] = (
                    val  # Avoid memory allocation by directly placing all outputs to the input dict
                )
            return data_batch
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise

    def _postprocess(self, data_batch: dict) -> dict:
        """
        Postprocess the model's prediction.

        Args:
            data_batch (dict): The output from the model.

        Returns:
            dict: Postprocessed prediction.
        """
        for key in self.output_specs["metric"]:
            try:
                data_batch[key] = data_batch[key].mean()
            except:
                # Sometime the metric might not be computed, e.g. during training the val metric
                pass
        return data_batch

    def _postprocess_after_optim(self, data_batch: dict) -> dict:
        """
        Additional postprocessing after optimizer.step

        Args:
            data_batch (dict): The postprocessed prediction.

        Returns:
            dict: Further processed prediction.
        """
        try:
            # Implement additional postprocessing steps after optimization
            return data_batch
        except Exception as e:
            logging.error(f"Error in postprocessing after optimization: {e}")
            raise

    def _dataparallel_postprocess(self, data_batch) -> dict:
        """
        Post-processes the data batch collected from multiple devices in a data parallel setting.

        In a data parallel environment, this function averages the loss and metric values across all devices. This is crucial for consistent and meaningful aggregation of these values when the model is trained or evaluated in parallel across multiple GPUs or other hardware.

        Parameters:
            - data_batch: A dictionary containing various keys representing different types of data (like loss, metrics, etc.). The values could be tensors or lists of tensors, depending on whether they are aggregated across devices.

        Returns:
            - A dictionary with the same structure as `data_batch`, but with loss and metric values averaged across devices.
        """
        if self.__dataparallel_flag__:
            for key in data_batch.keys():
                if key.endswith("loss") or key in self.output_specs["metric"]:
                    if isinstance(data_batch[key], list):
                        for idx in range(len(data_batch[key])):
                            data_batch[key][idx] = data_batch[key][idx].mean()
                    else:
                        data_batch[key] = data_batch[key].mean()
        return data_batch

    def _detach_before_return(self, data_batch: dict) -> dict:
        """
        Detach a tensor before returning, to remove it from the computation graph.
        """
        for key, val in data_batch.items():
            if isinstance(key, dict):
                data_batch[key] = self._detach_before_return(val)
            if isinstance(val, torch.Tensor):
                data_batch[key] = val.detach()
        return data_batch

    def zero_grad(self) -> None:
        """
        Clears the gradients of all optimized parameters for each optimizer in the model.

        This function iterates through all optimizers stored in the `optimizer_dict` attribute.
        It calls the `zero_grad()` method on each optimizer to reset the gradients.
        This is a necessary step in the training loop before performing a backward pass
        since PyTorch accumulates gradients on subsequent backward passes.
        """
        for optimizer_config in self.optimizer_dict.keys():
            self.optimizer_dict[optimizer_config].zero_grad()
        return

    def optimizers_step(self) -> None:
        """
        Executes the step function for all optimizers in the model.

        This function iterates through each optimizer stored in the `optimizer_dict` attribute
        and calls the `step()` method on them. The `step()` method updates the parameters of the
        model based on the gradients computed during the `backward()` pass in the training loop.
        This method should be called after `backward()` and any gradient manipulation (like clipping).
        """
        for optimizer_config in self.optimizer_dict.keys():
            self.optimizer_dict[optimizer_config].step()
        return

    def train_batch(self, data_batch: tuple, visualize=False) -> dict:
        """
        Train the model on a batch of data.

        Args:
            data_batch (tuple): The batch data for training.
            visualize (bool): A flag indicating whether to include additional visualization data in the output.

        Returns:
            dict: The training result.
        """
        try:
            # Preprocessing step
            data_batch = self._preprocess(data_batch, visualize)

            # Set training mode
            self.set_train()
            self.zero_grad()
            self.neural_network.zero_grad()  # Also remove some networks that is not in the optimizer list

            # Propagation of the preprocessed data throught the neural module
            data_batch = self._propagate(data_batch, visualize)

            # Postprocessing
            data_batch = self._postprocess(data_batch)

            # Clip loss
            if self.loss_clip > 0.0:
                if abs(data_batch["batch_loss"]) > self.loss_clip:
                    logging.warning(
                        f"Loss Clipped from {abs(data_batch['batch_loss'])} to {self.loss_clip}"
                    )
                data_batch["batch_loss"] = torch.clamp(
                    data_batch["batch_loss"], -self.loss_clip, self.loss_clip
                )

            # Backward pass to compute gradients
            data_batch["batch_loss"].backward()

            # Clip gradient
            if self.grad_clip > 0:
                for key in self.neural_network.network_dict.keys():
                    torch.nn.utils.clip_grad_norm_(
                        self.neural_network.network_dict[key].parameters(),
                        self.grad_clip,
                    )

            # Update model parameters
            self.optimizers_step()

            # Final postprocessing step
            data_batch = self._postprocess_after_optim(data_batch)
            data_batch = self._detach_before_return(data_batch)
            self.train_epoch += 1
            self.epoch += 1
            if self.auto_lr_adjust:
                self.adjust_learning_rate(self.train_epoch)

            return data_batch

        except Exception as e:
            logging.error(f"Error in training batch: {e}")
            raise

    def val_batch(self, data_batch: tuple, visualize=False) -> dict:
        """
        Validate the model on a batch of data.

        Args:
            data_batch (tuple): The batch data for validation.
            visualize (bool): A flag indicating whether to include additional visualization data in the output.

        Returns:
            dict: The validation result.
        """
        # Preprocessing step
        data_batch = self._preprocess(data_batch, visualize)

        # Set evaluation mode
        self.set_eval()

        # Propagation of the preprocessed data throught the neural module
        with torch.no_grad():
            data_batch = self._propagate(data_batch, visualize)

        # Postprocessing
        data_batch = self._postprocess(data_batch)
        data_batch = self._dataparallel_postprocess(data_batch)
        data_batch = self._postprocess_after_optim(data_batch)
        data_batch = self._detach_before_return(data_batch)
        self.val_epoch += 1
        self.epoch += 1

        return data_batch

    def model_resume(
        self,
        checkpoint,
        is_initialization,
        network_name=None,
        loading_ignore_key=[],
        strict=True,
    ) -> None:
        if "epoch" in checkpoint.keys() and "batch" in checkpoint.keys():
            state_dict = {}
            logging.info("Load from ep {}".format(checkpoint["epoch"]))
            self.train_epoch = checkpoint["epoch"]
            self.val_epoch = 0
            self.epoch = self.train_epoch + self.val_epoch
            for key, val in checkpoint["model_state_dict"].items():
                if key.startswith("module."):
                    name = ".".join(key.split(".")[1:])
                else:
                    name = key
                ignore_flag = False
                for ignore_key in loading_ignore_key:
                    if ignore_key in name:
                        logging.warning(
                            f"Ignore checkpoint {name} because set ignore key {ignore_key}"
                        )
                        ignore_flag = True
                        break
                if ignore_flag:
                    continue
                state_dict[name] = val
            checkpoint["model_state_dict"] = state_dict
            if not is_initialization or network_name == ["all"]:
                self.neural_network.load_state_dict(
                    checkpoint["model_state_dict"], strict=strict
                )
                if self.optimizer_dict != {}:
                    for key, val in checkpoint["optimizers_state_dict"]:
                        self.optimizer_dict[key].load_state_dict(val)
                        # Send to device
                        for state in self.optimizer_dict[key].state.values():
                            for _k, _v in state.items():
                                if torch.is_tensor(_v):
                                    state[_k] = _v.to(self.device, dtype=torch.float32)
            else:
                if network_name is not None:
                    prefix = ["network_dict." + name for name in network_name]
                    restricted_model_state_dict = {}
                    for key, val in checkpoint["model_state_dict"].items():
                        for pf in prefix:
                            if key.startswith(pf):
                                restricted_model_state_dict[key] = val
                                break
                    checkpoint["model_state_dict"] = restricted_model_state_dict
                self.neural_network.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
        elif "df" in checkpoint.keys():  # Loading the SDFusion model
            key_mapping = {
                "img_enc": "img_enc",
                "img_linear": "img_linear",
                "txt_enc": "txt_enc",
                "graph_enc": "bb_enc",
                "diffusion_net": "df",
                "denoiser": "df"
            }
            for key in self.neural_network.network_dict.keys():
                ckpt_key = key_mapping[key]

                self.neural_network.network_dict[key].load_state_dict(
                    checkpoint[ckpt_key]
                )

            # for key, val in checkpoint["optimizers_state_dict"]:
            #    self.optimizer_dict[key].load_state_dict(checkpoint['opt'])
            # # Send to device
            # for state in self.optimizer_dict[key].state.values():
            #     for _k, _v in state.items():
            #         if torch.is_tensor(_v):
            #             state[_k] = _v.to(self.device, dtype=torch.float32)

        else:
            self.neural_network.network_dict.load_state_dict(checkpoint)
        return

    def save_checkpoint(self, filepath: str, additional_dict: dict = None) -> None:
        """
        Save the current model state as a checkpoint.

        Args:
            checkpoint_path (str): The path to save the checkpoint.

        Raises:
            IOError: If the checkpoint cannot be saved.
        """
        try:
            save_dict = {
                "model_state_dict": (
                    self.neural_network.module.state_dict()
                    if self.__dataparallel_flag__
                    else self.neural_network.state_dict()
                ),
                "optimizers_state_dict": [
                    (k, opti.state_dict()) for k, opti in self.optimizer_dict.items()
                ],
            }

            if additional_dict is not None:
                for key, val in additional_dict.items():
                    save_dict[key] = val

            torch.save(save_dict, filepath)

        except IOError as e:
            logging.error(f"Error saving checkpoint: {e}")
            raise

        return

    def to_gpus(self) -> None:
        """
        Distribute the neural network model across available GPUs to enable parallel processing.

        This method checks the number of available GPU devices. If more than one GPU is available,
        it wraps the neural network model with `torch.nn.DataParallel`, which parallelizes the model
        across multiple GPUs. This helps in speeding up the training process by utilizing multiple GPUs
        for computation. If only one GPU is available, it sets a flag indicating that DataParallel is not used.
        Finally, the method ensures the model is moved to GPU memory by calling `.to(self.device)` on the neural network
        model, preparing it for GPU-based computations.
        """

        # By default, only one GPU is available: do not use DataParallel
        self.__dataparallel_flag__ = False

        os_type = platform.system()
        if os_type == "Linux" or os_type == "Windows":
            # Check if more than one GPU is available
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                # If yes, use DataParallel to distribute the model across GPUs
                self.neural_network = torch.nn.DataParallel(self.neural_network)
                # Set the flag indicating that DataParallel is being used
                self.__dataparallel_flag__ = True

        # Move the model to GPU memory
        self.neural_network.to(self.device, dtype=torch.float32)

        return

    def set_train(self) -> None:
        """
        Set the module into "training" mode.
        This mode is particularly important because it affects the behavior of certain layers and functions that have distinct behaviors during training and evaluation (inference) phases.
        Certain layers in a neural network behave differently during training and evaluation. For instance:
            Dropout Layers: During training, dropout randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
                            This helps prevent overfitting. During evaluation, dropout is disabled (i.e., it does nothing).
            Batch Normalization Layers: During training, these layers normalize the input using the mean and variance of the current batch.
                                        During evaluation, they use the running estimates of these statistics, which were computed during training.
        """
        self.neural_network.train()
        return

    def set_eval(self) -> None:
        """
        Set the module into "evaluation" mode.
        Certain layers in a neural network behave differently during training and evaluation. For instance:
            Dropout Layers: During training, dropout randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
                            This helps prevent overfitting. During evaluation, dropout is disabled (i.e., it does nothing).
            Batch Normalization Layers: During training, these layers normalize the input using the mean and variance of the current batch.
                                        During evaluation, they use the running estimates of these statistics, which were computed during training.
        """
        self.neural_network.eval()
        return
