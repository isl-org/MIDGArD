"""
Taken from https://github.com/JiahuiLei/NAP/tree/main/logger
"""

import pandas as pd
from .base_logger import BaseLogger
import os
import logging


class XLSLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, config) -> None:
        """
        XLS Logger constructor initializing from the BaseLogger, setting up logging paths and structures for Excel.
        """
        super().__init__(tb_logger, log_path, config)
        self.NAME = "xls"
        # Ensuring the log path exists.
        os.makedirs(self.log_path, exist_ok=True)
        self.pd_container = dict()  # Initializing a container to keep dataframes.
        self.current_epoch = 1  # Initializing current epoch.
        self.current_phase = "INIT"  # Initializing current phase.
        return

    def log_batch(self, batch):
        """
        Processes and logs the current batch's data to the pd_container for later export to Excel.

        :param batch: The batch containing data and meta information to log.
        """
        # Only proceed if the batch is relevant for the XLS logger (based on its output_parser).
        if self.NAME not in batch["output_parser"]:
            return

        keys_list = batch["output_parser"][self.NAME]
        if not keys_list:  # If keys list is empty, do nothing.
            return

        # Update the current state based on the batch info
        data = batch["data"]
        self.current_epoch = batch["epoch"]
        self.current_phase = batch["phase"]
        meta_info = batch["meta_info"]

        for sheet_key in keys_list:
            if sheet_key not in data:
                continue  # Skip if the sheet key doesn't exist in data.

            kdata = data[sheet_key]
            assert isinstance(
                kdata, dict
            ), "Data under each sheet must be a dictionary."

            # Initialize the pd_container for this sheet if it is not present.
            if sheet_key not in self.pd_container:
                self.pd_container[sheet_key] = pd.DataFrame()

            add_list = list()
            for ii in range(len(meta_info["viz_id"])):
                row_data = {k: v[ii] for k, v in kdata.items()}
                row_data["viz_id"] = meta_info["viz_id"][ii]
                add_list.append(row_data)

            self.pd_container[sheet_key] = pd.concat(
                [self.pd_container[sheet_key], pd.DataFrame(add_list)],
                ignore_index=True,  # Ensure the index does not repeat.
            )

    def log_phase(self) -> None:
        """
        Finalizes logging for the phase by writing the data to an Excel file and resetting the container.
        """
        for sheet_name, df in self.pd_container.items():
            if df.empty:
                continue  # Skip if the dataframe is empty.

            # Attempt to add a row with the mean values at the beginning of the dataframe.
            try:
                mean_row = df.mean(axis=0)
                self.pd_container[sheet_name] = pd.concat(
                    [mean_row.to_frame().T, df], ignore_index=False
                )
            except Exception as e:
                logging.warning(f"XLS logger add mean to head failed: {e}")

            # Write the dataframe to an Excel file.
            file_name = f"{sheet_name}_{self.current_epoch}_{self.current_phase}.xls"
            file_path = os.path.join(self.log_path, file_name)
            df.to_excel(file_path, index=False)

            # Reset the data container after logging.
            self.pd_container[sheet_name] = pd.DataFrame()
