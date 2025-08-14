"""Script used to train the midgard 2D VQVAE."""

import argparse
import sys

from scripts.utils.trainer import Trainer
from core.models.structure_generator.autoencoder.image_autoencoder import ImageVQVAE
from core.dataset.image_dataset import ImageDataset


def parse_batch_fn(batch, epoch, mode):
    """
    Optionally parse or modify each batch before the model processes it.

    Args:
        batch (Any):       The batch data. Typically a list or dictionary.
        epoch (int):       Current epoch number in the training loop.
        mode (str):        The mode string (e.g., 'train', 'val').

    Returns:
        Any: The modified batch.
    """
    # Example: If batch is a list of dictionaries, you can add the epoch info:
    # Ensuring the first element in the list has a dict
    if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
        batch[0]["epoch"] = epoch
    return batch


def main(argv) -> None:
    parser = argparse.ArgumentParser(description="Train an image VQVAE.")
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the random number generator."
    )
    args = parser.parse_args(argv)

    # Initialize the Trainer
    trainer = Trainer(
        args,
        model_cls=ImageVQVAE,
        dataset_cls=ImageDataset,
        parse_batch_fn=parse_batch_fn,
    )

    # Start training
    trainer.train()


# Main execution logic
if __name__ == "__main__":
    main(sys.argv[1:])
