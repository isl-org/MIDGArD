"""
Taken from https://github.com/JiahuiLei/NAP/tree/main/logger
"""

import os
from torch.utils.tensorboard import SummaryWriter
from .logger_meta import LOGGER_REGISTED, CheckpointLogger
from copy import deepcopy
import logging


class Logger(object):
    def __init__(self, config) -> None:
        """
        Initializes the Logger object.

        :param config: A dictionary containing the configuration for the logger.
        """
        # make a deep copy of the configuration to avoid accidental mutation
        self.config = deepcopy(config)

        # construct the path for the TensorBoard logs
        tb_path = os.path.join(
            config["base_directory"],
            config["output_directory"],
            config["logging"]["log_directory"],
            "tensorboardx",
        )

        # Initialize the TensorBoard writer
        self.tb_writer = SummaryWriter(tb_path)

        # Prepare the list of loggers based on configuration
        self.logger_list = self.compose(self.config["logging"]["loggers"])
        return

    def compose(self, names):
        """
        Compose the list of loggers from the given names.

        :param names: A list of logger names to be registered.
        :return: List of logger instances.
        """
        loggers_list = list()
        mapping = LOGGER_REGISTED
        for name in names:
            if name in mapping.keys():
                logger_instance = mapping[name](
                    self.tb_writer,
                    os.path.join(
                        os.path.join(
                            self.config["base_directory"],
                            self.config["output_directory"],
                            self.config["logging"]["log_directory"],
                            name,
                        )
                    ),
                    self.config,
                )
                loggers_list.append(logger_instance)
            else:
                raise Warning("Required logger " + name + " not found!")

        # log debug message to report the registered loggers
        logging.debug("Loggers [{}] registered".format(names))
        return loggers_list

    def log_phase(self) -> None:
        """
        Invokes the log_phase method on all registered loggers.
        """
        for logger in self.logger_list:
            logger.log_phase()

    def log_batch(self, batch) -> None:
        """
        Logs information about a batch using all registered loggers.

        :param batch: The batch to be logged.
        """
        for logger in self.logger_list:
            logger.log_batch(batch)

    def end_log(self) -> None:
        """
        Ends logging by calling log_phase on all registered loggers.
        """
        for logger in self.logger_list:
            logger.end_log()

    @property
    def model_logger(self) -> CheckpointLogger:
        """
        Returns the CheckpointLogger from the logger list.

        :return: An instance of the CheckpointLogger.
        :raise: RuntimeError if a CheckpointLogger is not found.
        """
        for logger in self.logger_list:
            if isinstance(logger, CheckpointLogger):
                return logger

        raise RuntimeError("Checkpoint logger not found")
