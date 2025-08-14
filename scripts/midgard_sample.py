"""Script used to sample the MIDGArD generative pipeline."""

import argparse
from copy import deepcopy
import logging
import numpy as np
import os
from rich.console import Console
from sklearn.neighbors import NearestNeighbors
import sys
import torch
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from core.dataset.graph_dataset import GraphDataset
from core.models.structure_generator.structure_generator import StructureGenerator
from core.models.structure_generator.autoencoder.image_autoencoder import ImageVQVAE
from core.utils.mesh_extractor import MeshExtractor
from core.utils.training_utils import load_config
from core.utils.mujoco_utils import generate_mujoco_scene_from_G, generate_mujoco_scene_from_VE
from scripts.utils.common import find_nn_database_mesh_and_update_G, extract_recon_mesh_for_nodes, viz_dir, load_latest_checkpoint


def get_articulation_model(config, checkpoint_directory):
    """
    Load the model defined in the configuration file and its last checkpoint.
    """
    # Denoising Diffusion Model:
    logging.info(f"Creating model.")
    model = StructureGenerator(config)
    logging.info(f"Model created.")

    # Load the desired checkpoint
    logging.info(f"Loading checkpoint...")
    checkpoint, training_episode = load_latest_checkpoint(checkpoint_directory)
    model.model_resume(checkpoint, is_initialization=True, network_name=["all"])
    logging.info(f"Checkpoint loaded.")

    # Copy model to GPU memory and prepare for evaluation
    model.to_gpus()
    model.set_eval()

    return model, training_episode


def get_vqvae_2d_model(config) -> ImageVQVAE:
    """
    Load the model defined in the configuration file and its last checkpoint.
    """
    # Autoencoder model:
    logging.info(f"Creating VQVAE 2D model...")
    model = ImageVQVAE(config)
    logging.info(f"Model created.")

    # Load the desired checkpoint
    logging.info(f"Loading checkpoint...")
    ckpt_fn = os.path.join(
        config["base_directory"],
        config["output_directory"],
        config["structure_generator"]["image_autoencoder"]["checkpoint_path"],
    )
    checkpoint = torch.load(ckpt_fn, map_location="cpu")
    model.model_resume(checkpoint, is_initialization=True, network_name=["all"])
    logging.info("Loaded image decoder weights from %s", ckpt_fn)

    # Copy model to GPU memory and prepare for evaluation
    model.to_gpus()

    # Set in evaluation mode to save memory
    model.set_eval()

    return model


def get_sdfusion_model(config):
    """
    Load the model defined in the configuration file and its last checkpoint.
    """
    from core.models.shape_generator.shape_generator import (
        StructureGenerator as SDFusion_Model,
    )

    # Autoencoder model:
    logging.info(f"Creating model.")
    model = SDFusion_Model(config)
    logging.info(f"Model created.")

    # Load the desired checkpoint
    logging.info(f"Loading checkpoint...")

    checkpoint_fn = config["ckpt"]
    map_fn = lambda storage, loc: storage
    logging.info("Loading checkpoint {}".format(checkpoint_fn))
    checkpoint = torch.load(checkpoint_fn, map_location=map_fn)
    logging.info("Checkpoint {} Loaded".format(checkpoint_fn))
    episode = 120000

    model.model_resume(checkpoint, is_initialization=True, network_name=["all"])
    logging.info(f"Checkpoint loaded.")

    # Copy model to GPU memory and prepare for evaluation
    model.to_gpus()
    model.set_eval()

    return model, episode


def generate_articulated_assets(
    config,
    eval_directory,
    checkpoint_directory,
    batch_size,
    use_hard_node_mask=False,
    sample_image_base_path=None,
    device=torch.device("cpu"),
):
    """
    Inference trained model to generate objects.
    Generates two folders under `eval_directory`: `G` saved all generated graph objects. `Viz` have some gifs visualization.
    """

    # Load dataset
    logging.info(f"Loading dataset.")
    dataset = GraphDataset(config, mode="train")
    logging.info(f"Dataset loaded.")

    if config.get("use_categorical_asset", False):
        config["asset_categories"] = dataset.embedding_model_cats
    if config.get("use_categorical_body", False):
        config["body_categories"] = dataset.embedding_body_types

    # Prepare database
    logging.info(f"Prepare database.")
    use_cat_body = config.get("use_categorical_body", False)
    use_cat_asset = config.get("use_categorical_asset", False)
    use_2d_latent = config.get("use_node_2D_preencoded_latent", False)

    if use_2d_latent:
        logging.info("Loading VQVAE_2D model...")
        vqvae_2d_model = get_vqvae_2d_model(config)
        logging.info("VQVAE_2D model loaded!")

    mesh_names = dataset.training_embedding_index
    training_node_features_scale = dataset.node_features_scale.clone()
    if use_cat_body:
        training_node_features_scale = np.concatenate(
            [
                np.ones(len(dataset.embedding_body_types)),
                training_node_features_scale,
            ],
            axis=-1,
        )
    if use_cat_asset:
        training_node_features_scale = np.concatenate(
            [
                np.ones(len(dataset.embedding_model_cats)),
                training_node_features_scale,
            ],
            axis=-1,
        )
    if use_2d_latent:
        database = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
            dataset.training_embedding_2d
        )
        training_node_features_scale = np.concatenate(
            [training_node_features_scale, dataset.embedding_scale_2d], axis=-1
        )

    training_node_features_scale = (
        torch.from_numpy(training_node_features_scale).float().to(device, dtype=torch.float32)
    )
    training_edge_features_scale = dataset.edge_features_scale.clone()
    training_edge_features_scale = (
        torch.from_numpy(training_edge_features_scale).float().to(device, dtype=torch.float32)
    )

    logging.info(f"Database ready.")

    # Load model
    model, training_episode = get_articulation_model(config, checkpoint_directory)

    # Start the generation process
    logging.info("Init generation process...")
    num_nodes = model.max_nodes
    noise_node_features = torch.randn(
        batch_size, num_nodes, model.feature_size["node"]
    ).to(device)
    noise_edge_features = torch.randn(
        batch_size, (num_nodes * (num_nodes - 1)) // 2, model.feature_size["edge"]
    ).to(device)
    if use_hard_node_mask:
        random_n = torch.randint(low=2, high=num_nodes + 1, size=(batch_size,))
        node_mask = torch.arange(num_nodes)[None, :] < random_n[:, None]
        node_mask = node_mask.float().to(device)
        noise_node_features[..., 0] = (
            node_mask  # set the noisy first channel to gt node_mask
        )
    else:
        node_mask = None

    logging.info("Start structure generation process...")

    # Call the articulated asset generation routine
    gen_node_features, gen_edge_features = model.neural_network.generate(
        noise_node_features,
        noise_edge_features,
        training_node_features_scale,
        training_edge_features_scale,
        node_mask=node_mask,
    )

    logging.info("Structure generation process completed.")

    # Save generated outputs (slow because it extracts the mesh with MonteCarlo)
    cates = config["dataset"]["cates"]
    destination_directory = os.path.join(
        eval_directory, f"G/K_{num_nodes}_cate_{''.join(cates)}_{training_episode}"
    )
    os.makedirs(destination_directory, exist_ok=True)
    
    destination_viz_directory = os.path.join(
        eval_directory, f"Viz/K_{num_nodes}_cate_{''.join(cates)}_{training_episode}"
    )
    os.makedirs(destination_viz_directory, exist_ok=True)
    
    destination_retrieval_directory = os.path.join(
        eval_directory,
        f"G/K_{num_nodes}_cate_{''.join(cates)}_{training_episode}_retrieval",
    )
    os.makedirs(destination_retrieval_directory, exist_ok=True)

    destination_retrieval_viz_directory = os.path.join(
        eval_directory,
        f"Viz/K_{num_nodes}_cate_{''.join(cates)}_{training_episode}_retrieval",
    )
    os.makedirs(destination_retrieval_viz_directory, exist_ok=True)
    
    mujoco_xml_directory = os.path.join(eval_directory, "MuJoCo")
    os.makedirs(mujoco_xml_directory, exist_ok=True)

    logging.info("Saving generated assets.")
    
    graphs = []
    fns = []
    
    # Process each item in the generated data
    for i in tqdm(range(gen_node_features.shape[0])):
        # Convert tensors to numpy arrays
        v, e = gen_node_features[i].cpu().numpy(), gen_edge_features[i].cpu().numpy()
        
        # Process and save the generated graph
        G = get_G_from_VE(
                v,
                e,
                configuration=model.ablations,
                feature_size=model.feature_size,
                feature_index=model.feature_index,
                body_categories=config.get("body_categories", None),
                asset_categories=config.get("asset_categories", None),
            )
        G = extract_recon_mesh_for_nodes(G, None, device)
        graphs.append(G)
        fns.append(f"gen_{i}")
        fn = os.path.join(destination_directory, f"gen_{i}.json")
        write_json(nx.adjacency_data(G), fn)
    
    logging.info("Generated assets saved.")

    # Generate MuJoCo XML files
    logging.info("Loading SDFusion model...")
    mesh_extractor = MeshExtractor()
    sdfusion_model, _ = get_sdfusion_model(config)
    logging.info("SDFusion model loaded!")
    
    # List all json files in the source directory
    fn_list = [
        os.path.basename(f) for f in os.listdir(destination_directory) if f.endswith(".json")
    ]
    
    # Load each graph and prepare parameters for mesh & .xml generation
    logging.info("Generating MuJoCo files...")
    
    for graph, fn in zip(graphs, fns):
        generate_mujoco_scene_from_G(
            graph_directory=destination_directory,
            output_directory=mujoco_xml_directory,
            asset_name=fn,
            graph=graph,
            use_graph_mesh=False,
            embedding_model_cats=dataset.embedding_model_cats,
            embedding_body_types=dataset.embedding_body_types,
            mesh_extractor=mesh_extractor,
            model=sdfusion_model,
            vqvae_2d_model=vqvae_2d_model if sample_image_base_path is None else None,
            sample_image_base_path=sample_image_base_path,
            device=device,
        )
        print(f"Finished generating {fn[:-4]}")
        
    logging.info("MuJoCo files generated.")
    
    # Rendering the generated assets
    logging.info("Rendering...")
    viz_dir(
        destination_directory,
        destination_viz_directory,
        max_viz=40,
        n_threads=os.cpu_count(),
    )
    viz_dir(
        destination_retrieval_directory,
        destination_retrieval_viz_directory,
        max_viz=40,
        n_threads=os.cpu_count(),
    )
    logging.info("Assets rendered.")

    return destination_directory, destination_retrieval_directory


def main(argv) -> None:
    matplotlib_axes_logger.setLevel("ERROR")
    logger = logging.getLogger("trimesh")
    logger.setLevel(logging.ERROR)

    console = Console()
    parser = argparse.ArgumentParser(description="Sample the midgard generative model.")
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the random number generator."
    )
    parser.add_argument(
        "--experiment_folder",
        default="graph_diffusion_experiment",
        help="Results and related content will be placed in output/experiment_folder.",
    )
    parser.add_argument(
        "--gen_directory",
        default="../output/generated",
        help="Where to save the generated assets.",
    )
    parser.add_argument(
        "--sample_image_base_path",
        default=None,
        type=str,
        help="where to sample images from",
    )
    parser.add_argument("--batch_size", default=10, type=int)
    args = parser.parse_args(argv)

    # Set the random seed
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
    num_states = args.n_states
    num_pcl = args.n_pcl
    batch_size = args.batch_size

    # Check if output directory exists. If it doesn't, then abort.
    output_directory = os.path.join(
        config["base_directory"], config["output_directory"]
    )
    assert os.path.exists(
        output_directory
    ), "The directory supposed to contain the experiment results does not exist. Aborting..."

    # Check if checkpoint directory exists. If it doesn't, then abort.
    checkpoint_directory = os.path.join(
        output_directory, "structure_generator", "checkpoint"
    )

    assert os.path.exists(
        checkpoint_directory
    ), "The directory supposed to contain the trained checkpoint does not exist. Aborting..."

    # Check if gen directory exists. If it doesn't, then create it.
    gen_directory = args.gen_directory
    os.makedirs(gen_directory, exist_ok=True)

    # Inference trained model to generate objects
    console.rule("[bold]Generating...")
    gen_dir_name, retrieval_dir_name = generate_articulated_assets(
        config,
        gen_directory,
        checkpoint_directory,
        args.batch_size,
        sample_image_base_path=args.sample_image_base_path,
    )

    console.rule("[bold]Done!")


# Main execution logic
if __name__ == "__main__":
    main(sys.argv[1:])
