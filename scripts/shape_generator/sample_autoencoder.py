"""Script used to sample a Shape Autoencoder."""

import argparse
import json
import logging
import numpy as np
import os
from PIL import Image
import random
import re
import sys
from rich.console import Console
import torch
import torchvision
from tqdm import tqdm
from core.utils.training_utils import load_config
from core.utils.image_processor import ImageProcessor, ImgProcCfg
from core.models.shape_generator.autoencoder.shape_autoencoder import ShapeVQVAE
from scripts.utils.common import load_latest_checkpoint


def get_model(config, checkpoint_directory):
    """
    Load the model defined in the configuration file and its last checkpoint.
    
    Args:
        config (dict): Configuration dictionary containing model parameters.
        checkpoint_directory (str): Directory where the model checkpoints are stored.
    
    Returns:
        model (torch.nn.Module): The loaded model.
        episode (int): The episode number of the loaded checkpoint.
    """
    # Autoencoder model:
    logging.info(f"Creating VQVAE model...")
    model = ShapeVQVAE(config)
    logging.info(f"Model created.")

    # Load the desired checkpoint
    logging.info(f"Loading checkpoint...")
    checkpoint, episode = load_latest_checkpoint(checkpoint_directory)
    model.model_resume(checkpoint, is_initialization=True, network_name=["all"])
    logging.info(f"Checkpoint loaded.")

    # Copy model to GPU memory and prepare for evaluation
    model.to_gpus()
    model.set_eval()

    return model, episode


def generate_latent_codes(config, gen_directory, checkpoint_directory, device) -> None:
    """
    Inference trained model to generate latent codes.
    Generates one file under `eval_directory`, containint the latent codes of all the assets of the dataset.
    
    Args:
        config (dict): Configuration dictionary containing model parameters.
        gen_directory (str): Directory where the generated latent codes will be saved.
        checkpoint_directory (str): Directory where the model checkpoints are stored.
        device (torch.device): Device to use for computation (CPU or GPU).
    
    Returns:
        None
    """
    # Data paths
    config_dict = config["dataset"]
    base_directory = config["base_directory"]
    data_directory = os.path.join(
        base_directory, config_dict["processed_data_directory"]
    )
    metadata_directory = os.path.join(base_directory, config_dict["metadata_directory"])

    # Load metadata
    asset_splits_path = os.path.join(
        metadata_directory, config_dict["articulated_splits"]
    )
    part_splits_path = os.path.join(metadata_directory, config_dict["part_splits"])
    info_path = os.path.join(metadata_directory, config_dict["dataset_info_file"])

    with open(info_path, "r") as f:
        info = json.load(f)

    with open(part_splits_path, "r") as f:
        mesh_id = json.load(f)  # Get a list of available meshes

    with open(asset_splits_path, "r") as f:
        asset_id = json.load(f)

    mesh_ids = mesh_id["train"] + mesh_id["val"] + mesh_id["test"]

    # Load model
    model, _ = get_model(config, checkpoint_directory)

    # Start the generation process
    logging.info("Init generation process...")

    assets_ids = {}
    for cat in asset_id.keys():
        assets_ids[cat] = (
            asset_id[cat]["train"] + asset_id[cat]["val"] + asset_id[cat]["test"]
        )

    # Create inverted index: category code -> keys in asset_id
    inverted_index = {}
    for key, codes in assets_ids.items():
        for code in codes:
            if code not in inverted_index:
                inverted_index[code] = key

    # Loop through each mesh id
    im_paths = []
    for mesh_id in mesh_ids:
        category_code, instance_number = mesh_id.split("_")

        if category_code in inverted_index:
            asset_type = inverted_index[category_code]
            im_paths.append(
                os.path.join(
                    data_directory,
                    asset_type,
                    str(category_code),
                    "images",
                    instance_number,
                    "1.png",
                )
            )

    logging.info("Start generation process...")
    
    # Build transform pipeline
    image_processor = ImageProcessor(ImgProcCfg())

    # Call the asset generation routine
    std_list = []
    latent_code_list = []
    valid_mask_list = []

    # Embedding generation
    for im_path in tqdm(im_paths):
        # Load and preprocess image
        image = image_processor(im_path, train=False).unsqueeze(0).to(device)

        # Compute the latent code
        with torch.no_grad():
            latent_code = model.neural_network.encode_no_quant(image).reshape(-1)

        latent_code_list.append(
            latent_code.detach()
            .reshape(
                -1,
            )
            .cpu()
        )
        valid_mask_list.append(True)  # TODO: change

    # Stack the tensors to create a [N, XXXX] tensor
    stacked_tensors = torch.stack(latent_code_list).cpu()

    # Standard deviation of the latent code (per dimension over the whole population)
    std_list = torch.std(stacked_tensors, dim=0).tolist()

    logging.info("Generation process completed.")

    # Save generated outputs
    output_file = os.path.join(gen_directory, f"latent_2d_data") + ".npz"

    logging.info("Saving generated assets.")
    np.savez(
        output_file,
        std=np.array(std_list),
        embedding=np.array(latent_code_list),
        valid_mask=np.array(valid_mask_list),
        ids=np.array(mesh_ids),
    )
    logging.info("Generated latent codes saved.")


def main(argv) -> None:
    """
    Main function to parse arguments and run the shape autoencoder sampling script.
    
    Args:
        argv (list): List of command line arguments.
    
    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console = Console()
    parser = argparse.ArgumentParser(description="Sample Shape Encoder.")
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the random number generator."
    )
    args = parser.parse_args(argv)

    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))
        # torch.backends.cudnn.benchmark = (
        #     True  # Enable cuDNN autotuner for performance optimization
        # )
        torch.backends.cudnn.deterministic = True
    elif torch.xpu.is_available():
        device = torch.device("xpu")
        torch.xpu.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.backends.mps.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))
    else:
        device = torch.device("cpu")

    # Parse the config file
    config = load_config(args.config_file)

    # Relevant parameters
    config["modes"] = ["train", "val", "test"]

    # Check if output directory exists. If it doesn't, then abort.
    output_directory = os.path.join(
        config["base_directory"], config["output_directory"]
    )
    assert os.path.exists(
        output_directory
    ), "The directory supposed to contain the experiment results does not exist. Aborting..."

    # Check if checkpoint directory exists. If it doesn't, then abort.
    checkpoint_directory = os.path.join(
        output_directory, config["logging"].get("log_directory", "logs"), "checkpoint"
    )
    assert os.path.exists(
        checkpoint_directory
    ), "The directory supposed to contain the trained checkpoint does not exist. Aborting..."

    # Check if gen directory exists. If it doesn't, then create it.
    gen_directory = os.path.join(output_directory, "gen")
    os.makedirs(gen_directory, exist_ok=True)

    # Inference trained model to generate objects
    console.rule("[bold]Generating...")
    generate_latent_codes(config, gen_directory, checkpoint_directory, device=device)

    console.rule("[bold]Done!")


# Main execution logic
if __name__ == "__main__":
    main(sys.argv[1:])
