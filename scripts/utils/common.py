from copy import deepcopy
import imageio
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from multiprocessing import Pool
import networkx as nx
import numpy as np
import open3d as o3d
import os
import re
from sklearn.neighbors import NearestNeighbors
import sys
import torch
from tqdm import tqdm
import trimesh

# Importing custom utility functions
from core.utils.data_utils import load_json, write_json, get_G_from_VE
from core.utils.visualization_utils import viz_G, viz_G_BB

sys.path.append(os.path.dirname(os.getcwd()))

matplotlib_axes_logger.setLevel("ERROR")


def load_latest_checkpoint(checkpoint_dir: str) -> tuple:
    """
    Load the latest checkpoint file from the given directory.

    Args:
    - checkpoint_dir: The directory containing the checkpoint files.

    Returns:
    - The loaded checkpoint, or None if no valid checkpoint is found.
    """
    # List all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    max_num = -1
    latest_checkpoint_file = None

    # Regular expression to identify checkpoint files by number
    pattern = re.compile(r"^(\d+)\.pt$")

    # Find the checkpoint with the highest number
    for file in checkpoint_files:
        match = pattern.match(file)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                latest_checkpoint_file = file

    map_fn = lambda storage, loc: storage

    # Load and return the latest checkpoint
    if latest_checkpoint_file:
        return (
            torch.load(
                os.path.join(checkpoint_dir, latest_checkpoint_file),
                map_location=map_fn,
            ),
            max_num,
        )
    else:
        print("No valid checkpoint found.")
        return None


def extract_recon_mesh_for_nodes(
    G: nx.Graph,
    extract_fn,
    device: torch.device = torch.device("cpu"),
) -> nx.Graph:
    """
    Extract reconstructed mesh for each node in a graph.

    Args:
    - G (nx.Graph): The graph with nodes containing mesh information.
    - extract_fn (function): Function to extract mesh from node attributes.
    - device (torch.device): Device to use for mesh extraction.

    Returns:
    - networkx.Graph: The graph with updated node attributes including the mesh.
    """
    # Iterate over each node in the graph
    for v, v_data in G.nodes(data=True):
        if "additional" not in v_data:
            continue

        # Use the 'additional' attribute to extract the mesh
        bbox = v_data["bbox"].copy()
        z = v_data["additional"][None, :]
        mesh = trimesh.primitives.Box(extents=2 * abs(bbox), mutable=True)
        mesh_centroid = mesh.bounds.mean(0)
        mesh.apply_translation(-mesh_centroid)
        nx.set_node_attributes(G, {v: {"mesh": mesh}})
    return G


def find_nn_database_mesh_and_update_G(
    G: nx.Graph,
    database: NearestNeighbors,
    mesh_names: list,
    data_directory: str,
    inverted_index: dict,
) -> nx.Graph:
    """
    Find nearest neighbor database mesh and update graph accordingly.

    Args:
    - G (networkx.Graph): Graph with nodes containing feature vectors.
    - database (NearestNeighbors): Pre-trained nearest neighbors model.
    - mesh_names (list): List of mesh names corresponding to database entries.
    - mesh_dir (str): Directory where mesh files are stored.

    Returns:
    - networkx.Graph: Updated graph with nearest neighbor mesh attached to nodes.
    """
    # Iterate over each node in the graph
    for v, v_data in G.nodes(data=True):
        if "2d_latent" in v_data:
            z = v_data["2d_latent"][None, :]
        elif "additional" in v_data:
            z = v_data["additional"][None, :]
        else:
            continue

        # Find nearest neighbor in the database
        _d, _ind = database.kneighbors(z, return_distance=True)
        _ind = int(_ind.squeeze(0))

        parts = mesh_names[int(_ind)].split("_")
        asset_code, mesh_index = parts[0], parts[1]
        asset_type = inverted_index[asset_code]
        file_path = os.path.join(
            data_directory,
            asset_type,
            str(asset_code),
            "manifold_meshes",
            mesh_index + ".obj",
        )
        gt_mesh = trimesh.load(file_path, force="mesh", process=False)
        mesh_centroid = gt_mesh.bounds.mean(0)
        gt_mesh.apply_translation(-mesh_centroid)
        bbox = v_data["bbox"].copy()
        scale = (
            2.0
            * np.linalg.norm(bbox)
            / np.linalg.norm(gt_mesh.bounds[1] - gt_mesh.bounds[0])
        )
        gt_mesh.apply_scale(scale)
        nx.set_node_attributes(G, {v: {"mesh": gt_mesh}})
    return G


def _viz_thread(p: tuple) -> None:
    """
    Visualizes a graph and saves it as an animated GIF.

    Parameters:
    p (tuple): A tuple containing the graph (G), the output filename (fn), and the number of frames (n_frames).

    Returns:
    None
    """
    G, fn, n_frames = p

    # Generate a visualization list of the graph
    viz_list = viz_G(G, cam_dist=3.0, viz_frame_N=n_frames, shape=(256, 256))

    print(f"finished render {fn}")
    # Save the visualization list as an animated GIF
    imageio.mimsave(fn, viz_list, fps=10)
    print(f"finished save {fn}")
    return


def viz_dir(
    src: str,
    dst: str,
    n_threads: int = os.cpu_count(),
    max_viz: int = 100,
    n_frames: int = 5,
) -> None:
    """
    Visualizes all graphs in a directory and saves them as animated GIFs.

    Parameters:
    src (str): Source directory containing json files of graphs.
    dst (str): Destination directory to save the animated GIFs.
    n_threads (int): Number of threads to use for parallel processing.
    max_viz (int): Maximum number of graphs to visualize.
    n_frames (int): Number of frames in each animated GIF.

    Returns:
    None
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(dst, exist_ok=True)

    # List all json files in the source directory
    fn_list = [f for f in os.listdir(src) if f.endswith(".json")]
    p_list = []

    # Load each graph and prepare parameters for visualization
    for fn in fn_list:
        G = nx.adjacency_graph(load_json(os.path.join(src, fn)))
        # nx.set_node_attributes(G, {v: {"mesh": gt_mesh}})
        
        viz_fn = os.path.join(dst, fn[:-5] + ".gif")
        p_list.append((G, viz_fn, n_frames))

    # Limit the number of visualizations if necessary
    if len(p_list) > max_viz:
        step = len(p_list) // max_viz
        p_list = p_list[::step]

    print(f"start rendering {len(p_list)} files...")

    # Use multiprocessing to visualize and save graphs in parallel
    with Pool(n_threads) as p:
        p.map(_viz_thread, p_list)

    return
