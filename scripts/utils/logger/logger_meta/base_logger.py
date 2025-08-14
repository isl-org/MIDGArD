"""
Taken from https://github.com/JiahuiLei/NAP/tree/main/logger
"""


class BaseLogger(object):
    def __init__(self, tb_logger, log_path, config) -> None:
        """
        Initialize the BaseLogger with configurations, logging paths, and a TensorBoard logger instance.

        :param tb_logger: An instance of a TensorBoard logger to log metrics for visual analysis.
        :param log_path: String representing the path where logs should be stored.
        :param config: Dictionary containing configurations for training and evaluation.
        """
        super().__init__()

        # Configuration dictionary containing training and evaluation settings
        self.config = config

        # A name identifier for this logger instance
        self.NAME = "base"

        # An instance of TensorBoard logger
        self.tb = tb_logger

        # Path to store log files
        self.log_path = log_path

        # Extract total number of epochs from the training configuration
        self.total_epoch = config["training"]["total_epoch"]

        # Extract batch size from the training configuration
        self.batch_size = config["training"]["batch_size"]

        # If the evaluation batch size is negative, fall back to training batch size.
        # Otherwise, use the specified evaluation batch size.
        if config["evaluation"]["batch_size"] < 0:
            self.eval_batch_size = config["training"]["batch_size"]
        else:
            self.eval_batch_size = config["evaluation"]["batch_size"]

    def log_phase(self) -> None:
        """
        Placeholder method to be implemented by subclasses for logging different phases like train, validate, or test.
        """
        pass

    def log_batch(self, batch) -> None:
        """
        Placeholder method to be implemented by subclasses for logging information specific to a batch.

        :param batch: The batch data that needs to be logged.
        """
        pass

    def end_log(self) -> None:
        """
        Placeholder method.
        """
        pass
