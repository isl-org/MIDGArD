import argparse
from joblib import Parallel, delayed
import json
import numpy as np
import os
import open3d as o3d
import shutil
import subprocess
import sys
from tqdm import tqdm


def generate_manifold_mesh(
    mesh_path,
    manifold_exec_path,
    target_number_of_triangles,
) -> None:
    """
    Generates a manifold version of a mesh by processing it with an external executable and post-processing.

    This function takes an input mesh (in formats such as .obj, .ply, .stl, .off, etc.), determines the output folder
    based on its location, and calls an external tool (specified by manifold_exec_path) to "manifoldize" the mesh.
    After the external processing, the function loads the resulting mesh using Open3D and applies a series of smoothing,
    decimation, and cleaning operations to achieve the target number of triangles and improve mesh quality.
    The final processed mesh is then saved back to disk.

    Args:
        mesh_path (str): Path to the input mesh file.
        manifold_exec_path (str): Path to the external manifold mesh generator executable.
        target_number_of_triangles (int): The maximum (target) number of triangles desired in the final mesh.

    Returns:
        None

    Raises:
        Exception: If the mesh file format is not supported.
    """
    # Mesh index without extension (ie. 5 if mesh is 38642_5.off)
    idx = os.path.splitext(os.path.basename(mesh_path))[0]

    # Folder that will contain the different images captured from this mesh
    path_parts = mesh_path.split(os.sep)
    path_index = path_parts.index("meshes") - 1
    path = os.sep.join(path_parts[: path_index + 1])
    output_manifold_folder = os.path.join(path, "images", idx)
    manifold_mesh_path = os.path.join(output_manifold_folder, str(idx) + ".obj")

    # Manifoldize mesh. Supported formats: .ply, .stl, .obj, .off, etc.
    if mesh_path.endswith(("obj", "ply", "stl", "3ds", "dae", "off")):
        result = subprocess.run(
            [manifold_exec_path] + [mesh_path, manifold_mesh_path, 50000],
            stdout=subprocess.PIPE,
            text=True,
        )
    else:
        raise Exception(f"Model format not supported: {mesh_path}")

    # Read the manifold mesh
    mesh = o3d.io.read_triangle_mesh(manifold_mesh_path)

    # Perform mesh post-processing operations
    mesh = (
        mesh.filter_smooth_taubin(number_of_iterations=100)
        .filter_smooth_laplacian(number_of_iterations=10)
        .simplify_quadric_decimation(
            target_number_of_triangles=target_number_of_triangles
        )
        .filter_smooth_simple(number_of_iterations=1)
        .remove_degenerate_triangles()
        .remove_duplicated_triangles()
        .remove_duplicated_vertices()
        .remove_non_manifold_edges()
        .remove_unreferenced_vertices()
        .compute_triangle_normals()
        .compute_vertex_normals()
    )

    # Save the post-processed final mesh
    o3d.io.write_triangle_mesh(manifold_mesh_path, mesh, write_vertex_normals=True)


def main(argv) -> None:
    """
    Main entry point for generating manifold meshes for a dataset.

    This function parses command-line arguments to obtain paths for the dataset directory, the manifold
    mesh generator executable, and the target number of triangles. It performs sanity checks on required
    directories, loads metadata regarding asset splits and mesh IDs, and builds an inverted index mapping.
    The function then copies raw mesh files into a structured directory (if needed) and uses parallel processing
    to run the generate_manifold_mesh function on each mesh.

    Args:
        argv (list of str): List of command-line arguments.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Mesh rendering routine.")
    parser.add_argument(
        "dataset_directory",
        type=str,
        help="Path to the directory containing the meshes.",
    )
    parser.add_argument(
        "--manifold_exec_path",
        default="",
        type=str,
        help="Path to the manifold mesh generator.",
    )
    parser.add_argument(
        "--target_number_of_triangles",
        default=10000,
        type=int,
        help="Max number of triangles in the output mesh.",
    )
    args = parser.parse_args(argv)

    # Sanity checks
    print(f"Browsing {args.dataset_directory}")
    data_directory = os.path.join(args.dataset_directory, "data")
    metadata_directory = os.path.join(args.dataset_directory, "metadata")
    assert os.path.exists(
        data_directory
    ), "The folder supposed to contain the data does not exist. Aborting..."
    assert os.path.exists(
        metadata_directory
    ), "The folder supposed to contain the metadata does not exist. Aborting..."

    # Load metadata
    asset_splits_path = os.path.join(metadata_directory, "articulated_splits.json")
    part_splits_path = os.path.join(metadata_directory, "part_splits.json")
    info_path = os.path.join(metadata_directory, "info.json")

    with open(info_path, "r") as f:
        info = json.load(f)

    with open(part_splits_path, "r") as f:
        mesh_id = json.load(f)

    with open(asset_splits_path, "r") as f:
        asset_id = json.load(f)

    assets_ids = {}
    for cat in asset_id.keys():
        assets_ids[cat] = (
            asset_id[cat]["train"] + asset_id[cat]["val"] + asset_id[cat]["test"]
        )

    # Initialize mesh_files_dict with the keys from asset_id and empty lists
    mesh_files_dict = {key: [] for key in assets_ids}

    # Create inverted index: inverted_index["8919"] = ["door"]
    inverted_index = {}
    for key, codes in assets_ids.items():
        for code in codes:
            if code not in inverted_index:
                inverted_index[code] = key

    # Loop through each mesh id
    mesh_ids = mesh_id["train"] + mesh_id["val"] + mesh_id["test"]
    for mesh_id in mesh_ids:
        category_code, instance_number = mesh_id.split("_")

        # If the mesh is in one of the considered categories
        if category_code in inverted_index:
            mesh_name = instance_number + ".off"
            raw_mesh_name = mesh_id + ".off"
            mesh_dir = os.path.join(
                data_directory, inverted_index[category_code], category_code, "meshes"
            )
            os.makedirs(mesh_dir, exist_ok=True)
            mesh_path = os.path.join(mesh_dir, mesh_name)
            if not os.path.exists(mesh_path):
                shutil.copy(
                    os.path.join(
                        args.dataset_directory,
                        "raw",
                        "partnet_mobility_graph_mesh",
                        raw_mesh_name,
                    ),
                    mesh_path,
                )

            mesh_files_dict[inverted_index[category_code]].append(mesh_path)
            manifold_dir = os.path.join(
                data_directory,
                inverted_index[category_code],
                category_code,
                "manifold_meshes",
            )
            os.makedirs(manifold_dir, exist_ok=True)

    for cat_id in info["all_cats"]:

        # Mesh files for a specific category of asset
        mesh_files = mesh_files_dict[cat_id]

        # Use joblib to create a pool of threads
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        with Parallel(n_jobs=os.cpu_count()) as executor:
            executor(
                delayed(generate_manifold_mesh)(
                    mesh_path, args.manifold_exec_path, args.target_number_of_triangles
                )
                for mesh_path in tqdm(mesh_files, desc="Processing meshes")
            )


# Main execution logic
if __name__ == "__main__":
    main(sys.argv[1:])
