"""Script used to evaluate the MIDGArD generative pipeline."""

from copy import deepcopy
import argparse
import logging
import networkx as nx
import numpy as np
import os
from rich.console import Console
from sklearn.neighbors import NearestNeighbors
import sys
import torch
from tqdm import tqdm
import trimesh
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from core.utils.training_utils import load_config
from core.dataset.graph_dataset import GraphDataset
from core.models.structure_generator.structure_generator import StructureGenerator
from core.models.structure_generator.autoencoder.image_autoencoder import ImageVQVAE
from core.utils.mesh_extractor import MeshExtractor
from core.utils.data_utils import load_json, write_json, get_G_from_VE
from core.utils.mujoco_utils import generate_mujoco_scene_from_G
from core.utils.visualization_utils import append_mesh_to_G
from scripts.utils.common import (
    find_nn_database_mesh_and_update_G,
    extract_recon_mesh_for_nodes,
    viz_dir,
    load_latest_checkpoint,
)
from scripts.utils.pcl_sampling_utils import sample_point_cloud
from scripts.utils.instantiation_distance_utils import compute_D_matrix
from scripts.utils.metrics_utils import eval_instantiation_distance


def get_articulation_model(config, checkpoint_directory):
    """
    Load the model defined in the configuration file and its last checkpoint.

    Args:
        config (dict): Configuration dictionary.
        checkpoint_directory (str): Path to the checkpoint directory.

    Returns:
        tuple: The loaded model and the training episode number.
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

    # Set in evaluation mode to save memory
    model.set_eval()

    return model, training_episode


def get_vqvae_2d_model(config) -> ImageVQVAE:
    """
    Load the model defined in the configuration file and its last checkpoint.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        ImageVQVAE: The loaded VQVAE model.
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


def get_sdfusion_model(config, checkpoint_directory):
    """
    Load the model defined in the configuration file and its last checkpoint.

    Args:
        config (dict): Configuration dictionary.
        checkpoint_directory (str): Path to the checkpoint directory.

    Returns:
        tuple: The loaded SDFusion model and the training episode number.
    """
    from core.models.shape_generator.shape_generator import ShapeGenerator

    # Autoencoder model:
    logging.info(f"Creating model.")
    model = ShapeGenerator(config)
    logging.info(f"Model created.")

    # Load the desired checkpoint
    logging.info(f"Loading checkpoint...")
    checkpoint_fn = os.path.join(checkpoint_directory, "last.pt")
    map_fn = lambda storage, loc: storage
    logging.info("Loading checkpoint {}".format(checkpoint_fn))
    checkpoint = torch.load(checkpoint_fn, map_location=map_fn)
    logging.info("Checkpoint {} Loaded".format(checkpoint_fn))
    episode = 120000

    model.model_resume(checkpoint, is_initialization=True, network_name=["all"])
    logging.info(f"Checkpoint loaded.")

    # Copy model to GPU memory and prepare for evaluation
    model.to_gpus()

    # Set in evaluation mode to save memory
    model.set_eval()

    return model, episode


def generate_articulated_assets(
    config,
    eval_directory,
    checkpoint_directory_structure_generator,
    checkpoint_directory_shape_generator,
    batch_size,
    use_hard_node_mask=True,
    sample_image_base_path=None,
    device=torch.device("cpu"),
):
    """
    Inference trained model to generate objects.
    Generates two folders under `eval_directory`: `G` saved all generated graph objects. `Viz` have some gifs visualization.

    Args:
        config (dict): Configuration dictionary.
        eval_directory (str): Directory to save the generated assets.
        checkpoint_directory_structure_generator (str): Path to the structure generator checkpoint directory.
        checkpoint_directory_shape_generator (str): Path to the shape generator checkpoint directory.
        batch_size (int): Batch size for generation.
        use_hard_node_mask (bool): Whether to use hard node mask or not.
        sample_image_base_path (str, optional): Path to sample images from. Defaults to None.
        device (torch.device, optional): Device to run the model on. Defaults to torch.device("cpu").

    Returns:
        tuple: Paths to the generated assets and retrieval directory, model configurations.
    """
    # Data paths
    config_dict = config["dataset"]
    base_directory = config["base_directory"]  # Defined in the training script
    data_directory = os.path.join(
        base_directory, config_dict["processed_data_directory"]
    )
    metadata_directory = os.path.join(base_directory, config_dict["metadata_directory"])
    codebook_directory = os.path.join(base_directory, config_dict["codebook_directory"])

    # Load metadata
    asset_splits_path = os.path.join(
        metadata_directory, config_dict["articulated_splits"]
    )
    part_splits_path = os.path.join(metadata_directory, config_dict["part_splits"])
    info_path = os.path.join(metadata_directory, config_dict["dataset_info_file"])

    info = load_json(info_path)
    mesh_id = load_json(part_splits_path)
    asset_id = load_json(asset_splits_path)

    # Retrieve and sort dataset files
    assets_ids = {}
    for cat in asset_id.keys():
        assets_ids[cat] = (
            asset_id[cat]["train"] + asset_id[cat]["val"] + asset_id[cat]["test"]
        )

    # Create inverted index: category code -> keys in asset_id
    inverted_index = {}
    for asset_type, codes in assets_ids.items():
        for code in codes:
            inverted_index[code] = asset_type

    # Load dataset
    logging.info(f"Loading dataset.")
    dataset = GraphDataset(config, mode="test")
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
    vqvae_2d_model = None
    if use_2d_latent:
        logging.info("Loading VQVAE_2D model...")
        vqvae_2d_model = get_vqvae_2d_model(config)
        logging.info("VQVAE_2D model loaded!")

    mesh_names = dataset.training_embedding_2d_ids
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
        torch.from_numpy(training_node_features_scale)
        .float()
        .to(device, dtype=torch.float32)
    )
    training_edge_features_scale = dataset.edge_features_scale.clone().to(
        device, dtype=torch.float32
    )
    # training_edge_features_scale = (
    #     torch.from_numpy(training_edge_features_scale).float().to(device, dtype=torch.float32)
    # )

    #################################################################################################
    #################################################################################################
    #################################################################################################

    # Load dataset
    config["dataset"]["modes"] = ["test"]
    test_dataset = GraphDataset(config, mode="test")

    # Load model
    model, training_episode = get_articulation_model(
        config, checkpoint_directory_structure_generator
    )

    # Start the generation process
    logging.info("Init generation process...")
    num_nodes = model.max_nodes
    num_instances = 2 #len(test_dataset)  # 385
    # noise_node_features = torch.randn(batch_size, num_nodes, model.feature_size["node"]).to(device, dtype=torch.float32)
    # noise_edge_features = torch.randn(batch_size, (num_nodes * (num_nodes - 1)) // 2, model.feature_size["edge"]).to(device, dtype=torch.float32)
    noise_node_features = torch.randn(
        num_instances, num_nodes, model.feature_size["node"]
    ).to(device, dtype=torch.float32)
    noise_edge_features = torch.randn(
        num_instances, (num_nodes * (num_nodes - 1)) // 2, model.feature_size["edge"]
    ).to(device, dtype=torch.float32)
    feature_index = model.feature_index

    # Use specififc features of the test dataset as starting points for the generation
    aggregated_node_features = []
    aggregated_edge_features = []
    aggregated_node_mask = []
    aggregated_edge_mask = []
    for data, _ in tqdm(test_dataset):
        known_node_features_i = data["V"]
        known_edge_features_i = data["E"]
        node_mask_i = torch.ones_like(data["V"])  # 1: uncond, 0: cond
        edge_mask_i = torch.ones_like(data["E"])  # 1: uncond, 0: cond

        if not model.ablations["use_node_orientation"]:
            # Remove the node rotation for now: assume that all the mesh rotations are set to 0.
            # o_i(1), sa_i(46) + sb_i(107) + bb_i(3) = 157
            known_node_features_i = torch.cat(
                [
                    known_node_features_i[..., : model.feature_index["node_r"]],
                    known_node_features_i[
                        ...,
                        model.feature_index["node_r"] + model.feature_size["node_r"] :,
                    ],
                ],
                -1,
            )  # Now node_features: [o_i(1), asset_label(46), part_label(107), bbox(3), t_gl(3)]

            node_mask_i = torch.cat(
                [
                    node_mask_i[..., : model.feature_index["node_r"]],
                    node_mask_i[
                        ...,
                        model.feature_index["node_r"] + model.feature_size["node_r"] :,
                    ],
                ],
                -1,
            )
        
        # node_mask_i[..., 0] *= 0
        # edge_mask_i[..., 0:feature_index["edge_j"]] *= 0
        # edge_mask_i[..., feature_index["edge_p"]:feature_index["edge_l"]] *= 0

        # if model.ablations["use_categorical_asset"]:
        node_mask_i[..., feature_index["node_sa"]:feature_index["node_sb"]] *= 0
        # if model.ablations["use_categorical_body"]:
        # node_mask_i[..., feature_index["node_sb"]:feature_index["node_bb"]] *= 0
        # #if model.ablations["use_node_2D_preencoded_latent"]:
        # node_mask_i[..., feature_index["node_2d"]:] *= 0
        # #if model.ablations["use_categorical_joint"]:
        # edge_mask_i[..., feature_index["edge_j"]:feature_index["edge_p"]] *= 0

        # node_mask_i[..., feature_index["node_bb"]:feature_index["node_r"]] *= 0
        # node_mask_i[..., feature_index["node_r"]:feature_index["node_t"]] *= 0
        # node_mask_i[..., feature_index["node_t"]:feature_index["node_2d"]] *= 0

        aggregated_node_features.append(known_node_features_i)
        aggregated_edge_features.append(known_edge_features_i)
        aggregated_node_mask.append(node_mask_i)
        aggregated_edge_mask.append(edge_mask_i)

    aggregated_node_features = aggregated_node_features[:num_instances]
    aggregated_edge_features = aggregated_edge_features[:num_instances]
    aggregated_node_mask = aggregated_node_mask[:num_instances]
    aggregated_edge_mask = aggregated_edge_mask[:num_instances]
        
    # Build masked test batch batch
    known_node_features = (
        torch.stack(aggregated_node_features, dim=0)
        .float()
        .to(device, dtype=torch.float32)
    )
    known_edge_features = (
        torch.stack(aggregated_edge_features, dim=0)
        .float()
        .to(device, dtype=torch.float32)
    )
    node_mask = (
        torch.stack(aggregated_node_mask, dim=0).float().to(device, dtype=torch.float32)
    )
    edge_mask = (
        torch.stack(aggregated_edge_mask, dim=0).float().to(device, dtype=torch.float32)
    )

    logging.info("Start structure generation process...")

    # Call the articulated asset generation routine with proper prior
    gen_node_features, gen_edge_features = model.neural_network.masked_generate(
        noise_node_features,
        noise_edge_features,
        training_node_features_scale,
        training_edge_features_scale,
        node_mask=node_mask,
        known_node_features=known_node_features,
        edge_mask=edge_mask,
        known_edge_features=known_edge_features,
    )

    # gen_node_features, gen_edge_features = model.neural_network.generate(
    #     noise_node_features,
    #     noise_edge_features,
    #     training_node_features_scale,
    #     training_edge_features_scale,
    # )

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

        # Process and save the retrieved graph
        G_r = deepcopy(G)
        G_r = find_nn_database_mesh_and_update_G(
            G_r, database, mesh_names, data_directory, inverted_index
        )
        fn = os.path.join(destination_retrieval_directory, f"gen_retrieval_{i}.json")
        write_json(nx.adjacency_data(G_r), fn)

    logging.info("Generated assets saved.")

    # Mesh extractor
    logging.info("Loading SDFusion model...")
    mesh_extractor = MeshExtractor()
    sdfusion_model, _ = get_sdfusion_model(config, checkpoint_directory_shape_generator)
    logging.info("SDFusion model loaded!")

    # List all json files in the source directory
    fn_list = [
        os.path.basename(f)
        for f in os.listdir(destination_directory)
        if f.endswith(".json")
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

    return (
        destination_directory,
        destination_retrieval_directory,
        model.ablations,
        model.feature_size,
        model.feature_index,
        model.body_categories,
        model.asset_categories,
    )


def generate_articulated_assets_gt(
    config,
    eval_directory,
    ablations,
    feature_size,
    feature_index,
    body_categories,
    asset_categories,
) -> str:
    """
    Save the ground truth objects for evaluation.

    Args:
        config (dict): Configuration dictionary.
        eval_directory (str): Directory to save the ground truth assets.
        ablations (dict): Ablation study configuration.
        feature_size (dict): Feature size configuration.
        feature_index (dict): Feature index configuration.
        body_categories (list): List of body categories.
        asset_categories (list): List of asset categories.

    Returns:
        str: Path to the directory containing the ground truth assets.
    """
    config["dataset"]["modes"] = ["test"]
    cates = config["dataset"]["cates"]
    num_nodes = config["dataset"]["max_nodes"]

    # Data paths
    config_dict = config["dataset"]
    base_directory = config["base_directory"]  # Defined in the training script
    data_directory = os.path.join(
        base_directory, config_dict["processed_data_directory"]
    )
    metadata_directory = os.path.join(base_directory, config_dict["metadata_directory"])
    codebook_directory = os.path.join(base_directory, config_dict["codebook_directory"])

    # Load metadata
    asset_splits_path = os.path.join(
        metadata_directory, config_dict["articulated_splits"]
    )
    part_splits_path = os.path.join(metadata_directory, config_dict["part_splits"])
    info_path = os.path.join(metadata_directory, config_dict["dataset_info_file"])

    info = load_json(info_path)
    mesh_id = load_json(part_splits_path)
    asset_id = load_json(asset_splits_path)

    # Retrieve and sort dataset files
    assets_ids = {}
    for cat in asset_id.keys():
        assets_ids[cat] = asset_id[cat]["test"]
        
    # Create inverted index: category code -> keys in asset_id
    inverted_index = {}
    for asset_type, codes in assets_ids.items():
        for code in codes:
            inverted_index[code] = asset_type
            mesh_directory = os.path.join(
                data_directory, asset_type, str(code), "manifold_meshes"
            )
            fn_list = [
                os.path.basename(f)
                for f in os.listdir(mesh_directory)
                if f.endswith(".obj")
            ]

            for m_id in fn_list:
                mesh_path = os.path.join(mesh_directory, str(m_id))
            
    # Load dataset
    logging.info(f"Loading dataset.")
    dataset = GraphDataset(config, mode="test")
    logging.info(f"Dataset loaded.")
    
    if config.get("use_categorical_asset", False):
        config["asset_categories"] = dataset.embedding_model_cats
    if config.get("use_categorical_body", False):
        config["body_categories"] = dataset.embedding_body_types

    destination_directory = os.path.join(
        eval_directory, f"G/K_{num_nodes}_cate_{''.join(cates)}_gt"
    )
    destination_viz_directory = os.path.join(
        eval_directory, f"Viz/K_{num_nodes}_cate_{''.join(cates)}_gt"
    )
    os.makedirs(destination_directory, exist_ok=True)
    os.makedirs(destination_viz_directory, exist_ok=True)

    # Save ground truth
    for data, meta in tqdm(dataset):
        fn = os.path.join(destination_directory, f"{meta['partnet-m-id']}.json")
        mesh_name_list = meta["mesh_name_list"]
        gt_mesh_list = []
        for mesh_name in mesh_name_list:
            parts = mesh_name.split("_")
            if len(parts) == 2:
                asset_code, mesh_index = parts[0], parts[1]
                asset_type = inverted_index[asset_code]

                gt_mesh_fn = os.path.join(
                    data_directory,
                    asset_type,
                    str(asset_code),
                    "manifold_meshes",
                    mesh_index + ".obj",
                )

                gt_mesh_list.append(trimesh.load(gt_mesh_fn, force="mesh"))

        V, E = data["V"], data["E"]
        V_scale, E_scale = data["V_scale"], data["E_scale"]
        V = V * V_scale  # Multiply back to the original scale
        E = E * E_scale  # Multiply back to the original scale
        v_mask = V[:, 0] > 0
        assert v_mask.sum() == len(gt_mesh_list)
        assert v_mask[: len(gt_mesh_list)].all()

        G = get_G_from_VE(
            V.cpu().numpy(),
            E.cpu().numpy(),
            configuration=ablations,
            feature_size=feature_size,
            feature_index=feature_index,
            body_categories=body_categories,
            asset_categories=asset_categories,
        )

        G = append_mesh_to_G(G, gt_mesh_list, key="mesh")  # with key mesh
        write_json(nx.adjacency_data(G), fn)

    # viz_dir(
    #     destination_directory,
    #     destination_viz_directory,
    #     max_viz=40,
    #     n_threads=os.cpu_count(),
    #     n_frames=7,
    # )

    return destination_directory


def compute_instantiation_distance(
    eval_directory: str,
    gt_directory: str,
    gen_directory: str,
    retrieval_directory: str,
    N_states_max=10,
    N_pcl_max=2048,
) -> None:
    """
    Computes the instantiation distance between different sets of point clouds.

    This function calculates the distance matrices between ground truth, generated, and retrieved point clouds to evaluate their similarity.

    Parameters:
    eval_directory (str): Directory where evaluation results will be saved.
    gt_directory (str): Directory containing ground truth point clouds.
    gen_directory (str): Directory containing generated point clouds.
    retrieval_directory (str): Directory containing retrieved point clouds.
    N_states_max (int): Maximum number of states (frames) to consider in the evaluation.
    N_pcl_max (int): Maximum number of points in each point cloud for the evaluation.

    Returns:
    None
    """
    # Directory to save the distance matrices
    save_directory = os.path.join(eval_directory, "ID_D_matrix")

    # Compute and save distance matrices for various combinations of point cloud sets
    compute_D_matrix(
        gen_dir=gt_directory,
        ref_dir=gt_directory,
        save_dir=save_directory,
        N_states_max=N_states_max,
        N_pcl_max=N_pcl_max,
    )
    compute_D_matrix(
        gen_dir=gen_directory,
        ref_dir=gt_directory,
        save_dir=save_directory,
        N_states_max=N_states_max,
        N_pcl_max=N_pcl_max,
    )
    compute_D_matrix(
        gen_dir=gen_directory,
        ref_dir=gen_directory,
        save_dir=save_directory,
        N_states_max=N_states_max,
        N_pcl_max=N_pcl_max,
    )
    compute_D_matrix(
        gen_dir=retrieval_directory,
        ref_dir=gt_directory,
        save_dir=save_directory,
        N_states_max=N_states_max,
        N_pcl_max=N_pcl_max,
    )
    compute_D_matrix(
        gen_dir=retrieval_directory,
        ref_dir=retrieval_directory,
        save_dir=save_directory,
        N_states_max=N_states_max,
        N_pcl_max=N_pcl_max,
    )


def main(argv) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    matplotlib_axes_logger.setLevel("ERROR")
    console = Console()
    parser = argparse.ArgumentParser(
        description="Evaluate the midgard generative model."
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the random number generator."
    )
    parser.add_argument(
        "--gen_directory",
        default="eval_cond",
        help="Results and related content will be placed in output/experiment_folder.",
    )
    parser.add_argument(
        "--sample_image_base_path",
        default=None,
        type=str,
        help="where to sample images from",
    )
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--n_states", default=50, type=int)
    parser.add_argument("--n_pcl", default=4096, type=int)
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
    checkpoint_directory_structure_generator = os.path.join(
        output_directory, "structure_generator", "checkpoint"
    )

    assert os.path.exists(
        checkpoint_directory_structure_generator
    ), "The directory supposed to contain the trained checkpoin for the structure generator does not exist. Aborting..."

    checkpoint_directory_shape_generator = os.path.join(
        output_directory, "shape_generator", "checkpoint"
    )

    assert os.path.exists(
        checkpoint_directory_shape_generator
    ), "The directory supposed to contain the trained checkpoint for the shape generator does not exist. Aborting..."

    # Check if eval directory exists. If it doesn't, then create it.
    eval_directory = os.path.join(output_directory, args.gen_directory)
    os.makedirs(eval_directory, exist_ok=True)
    # shutil.rmtree(eval_directory) # remove existing evals to rpevent data corruption from previous runs

    # Inference trained model to generate objects
    console.rule("[bold]Generating...")
    (
        gen_dir_name,
        retrieval_dir_name,
        ablations,
        feature_size,
        feature_index,
        body_categories,
        asset_categories,
    ) = generate_articulated_assets(
        config,
        eval_directory,
        checkpoint_directory_structure_generator,
        checkpoint_directory_shape_generator,
        batch_size,
        sample_image_base_path=args.sample_image_base_path,
        device=device,
    )

    # Save the ground truth objects for evaluation
    # console.rule("[bold]Generating ground truth...")
    gt_dir_name = generate_articulated_assets_gt(
        config,
        eval_directory,
        ablations=ablations,
        feature_size=feature_size,
        feature_index=feature_index,
        body_categories=body_categories,
        asset_categories=asset_categories,
    )

    # Sample point clouds from generated objects
    console.rule("[bold]Sampling point cloud...")
    use_categorical_joint = True  # ablations["use_categorical_joint"]
    # gen_dir_name = os.path.join(eval_directory, f"G/K_8_cate_all_10527")
    # retrieval_dir_name = os.path.join(eval_directory, f"G/K_8_cate_all_10527_retrieval")
    # gt_dir_name = os.path.join(eval_directory, f"G/K_8_cate_all_gt")

    sample_point_cloud(gt_dir_name, num_states, num_pcl, use_categorical_joint)
    sample_point_cloud(gen_dir_name, num_states, num_pcl, use_categorical_joint)
    sample_point_cloud(retrieval_dir_name, num_states, num_pcl, use_categorical_joint)

    # Compute the instantiation distance (very slow) based on the saved pcl files
    # console.rule("[bold]Conputing instantiation distance...")
    compute_instantiation_distance(
        eval_directory, gt_dir_name, gen_dir_name, retrieval_dir_name, 10, 2048
    )

    # Compute the metrics from the saved distance matrices
    console.rule("[bold]Computing metrics...")
    eval_instantiation_distance(eval_directory, gen_dir_name, gt_dir_name, 10, 2048)
    eval_instantiation_distance(
        eval_directory, retrieval_dir_name, gt_dir_name, 10, 2048
    )

    console.rule("[bold]Done.")


# Main execution logic
if __name__ == "__main__":
    main(sys.argv[1:])
