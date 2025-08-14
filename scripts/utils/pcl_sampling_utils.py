# sample saved pkl Graphs

from multiprocessing import Pool
import networkx as nx
import numpy as np
import open3d as o3d
import os
from random import shuffle
from tqdm import tqdm
from transforms3d.axangles import axangle2mat
import trimesh
import shutil

from core.utils.data_utils import load_json


def sample(
    pkl_fn: str,
    dst_fn: str,
    N_states: int = 100,
    N_PCL: int = 10000,
    use_categorical_joint=False,
) -> None:
    """
    Samples point clouds and poses from a graph representation stored in a json file.

    Parameters:
    pkl_fn (str): Path to the input json file containing the graph representation.
    dst_fn (str): Path to the output compressed .npz file where sampled data will be stored.
    N_states (int): Number of states (frames) to sample.
    N_PCL (int): Number of points in each point cloud to sample.

    Returns:
    None
    """
    # Load the graph from json file
    G = nx.adjacency_graph(load_json(pkl_fn))

    # Forward propagate through the graph to get mesh and pose lists
    mesh_list, pose_list = forward_G(
        G, N_frame=N_states, use_categorical_joint=use_categorical_joint
    )

    # Sample point clouds from the mesh
    # pcl_list = [sample_mesh(mesh, N_PCL) for mesh in mesh_list]
    pcl_list = [sample_tmesh(mesh, N_PCL) for mesh in mesh_list]

    # Stack pose and point cloud lists for saving
    pose_list = np.stack(pose_list, 0)  # N_states, N_parts, 4,4
    pcl_list = np.stack(pcl_list, 0)  # N_states, N_pcl, 3

    # Save the sampled data to a compressed .npz file
    np.savez_compressed(dst_fn, pcl=pcl_list, pose=pose_list)

    return


def sample_mesh(mesh: o3d.geometry.TriangleMesh, pre_sample_n: int) -> np.ndarray:
    """
    Samples a given number of points from the surface of an Open3D triangle mesh object.

    Parameters:
    mesh (o3d.geometry.TriangleMesh): The Open3D triangle mesh object to sample from.
    pre_sample_n (int): The number of points to sample.

    Returns:
    numpy.ndarray: An array of sampled points.
    """
    # Ensure the mesh has triangles
    if np.asarray(mesh.triangles).shape[0] == 0:
        return np.array([])

    # Compute the triangle areas if not already done
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    # Sample points from the surface of the mesh
    pcl = mesh.sample_points_poisson_disk(number_of_points=pre_sample_n)

    # Extract the point coordinates
    pcl = np.asarray(pcl.points, dtype=np.float16)

    return pcl


def sample_tmesh(tmesh, pre_sample_n):
    pcl, _ = trimesh.sample.sample_surface_even(tmesh, pre_sample_n * 2)
    while len(pcl) < pre_sample_n:
        _pcl, _ = trimesh.sample.sample_surface_even(tmesh, pre_sample_n * 2)
        pcl = np.concatenate([pcl, _pcl])
    pcl = pcl[:pre_sample_n]
    pcl = np.asarray(pcl, dtype=np.float16)
    return pcl


def screw_to_T(theta: float, d: float, l: np.ndarray, m: np.ndarray):
    """
    Converts screw parameters to a transformation matrix.

    Parameters:
    theta (float): The rotation angle.
    d (float): The displacement along the screw axis.
    l (numpy.ndarray): The direction vector of the screw axis.
    m (numpy.ndarray): The moment vector of the screw axis.

    Returns:
    numpy.ndarray: The resulting transformation matrix.
    """
    # Ensure the direction vector is normalized
    assert abs(np.linalg.norm(l) - 1.0) < 1e-4, "The vector is not normalized"

    # Calculate rotation and translation components
    R = axangle2mat(l, theta)
    t = (np.eye(3) - R) @ (np.cross(l, m)) + l * d

    # Construct the transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def forward_G(
    G: nx.Graph,
    N_frame: int = 100,
    mesh_key: str = "mesh",
    use_categorical_joint=False,
):
    """
    Propagates through a graph to extract mesh and pose information.

    Parameters:
    G (networkx.Graph): The graph representing the object.
    N_frame (int): Number of frames to generate.
    mesh_key (str): Key to access mesh data in graph nodes.

    Returns:
    list: A list of concatenated meshes for each frame.
    list: A list of transformation matrices for each part in each frame.
    """
    # Initialize lists to store mesh and pose data
    POSE, MESH = [], []

    # Check that the graph is a tree
    assert len(G.nodes) >= 2 and nx.is_tree(G)  # now only support tree viz

    # Now G is connected and acyclic
    num_nodes = [n for n in G.nodes]
    v_bbox = np.stack([d["bbox"] for nid, d in G.nodes(data=True)], 0)
    v_volume = v_bbox.prod(axis=-1) * 8
    root_vid = num_nodes[v_volume.argmax()]

    # Traverse the graph and calculate transformations for each node
    # The code inside this loop processes each node in the graph,
    # calculating transformation matrices and applying them to meshes.
    # It's quite complex and specific to the data structure and application.
    for _ in tqdm(range(N_frame)):
        node_traverse_list = [n for n in nx.dfs_preorder_nodes(G, root_vid)]
        T_rl_list = [np.eye(4)]  # p_root = T_rl @ p_link

        # Prepare the node pos
        for i in range(len(node_traverse_list) - 1):
            cid = node_traverse_list[i + 1]
            for e, e_data in G.edges.items():
                if cid in e:
                    # determine the direction by ensure the other end is a predessor in the traversal list
                    other_end = e[0] if e[1] == cid else e[1]
                    if node_traverse_list.index(other_end) > i:
                        continue
                    else:
                        pid = other_end

                    # T1: e_T_src_j1, T2: e_T_j2_dst
                    e_data = G.edges[e]
                    _T0 = e_data["T_src_dst"]
                    plucker = e_data["plucker"]
                    l, m = np.array(plucker[:3]), np.array(plucker[3:])
                    plim, rlim = e_data["plim"], e_data["rlim"]
                    if use_categorical_joint:
                        joint_label = e_data["joint_label"]
                        rlim, plim = resolve_range(joint_label, rlim, plim)

                    # random sample
                    # theta = np.linspace(*rlim, N_frame)[step]
                    # d = np.linspace(*plim, N_frame)[step]
                    theta = np.random.uniform(*rlim)
                    d = np.random.uniform(*plim)

                    _T1 = screw_to_T(theta, d, l, m)
                    T_src_dst = _T1 @ _T0
                    if pid == e_data["src"]:  # parent is src
                        T_parent_child = T_src_dst
                    else:  # parent is dst
                        T_parent_child = np.linalg.inv(T_src_dst)
                        # T_parent_child = T_src_dst
                    T_root_child = (
                        T_rl_list[node_traverse_list.index(pid)] @ T_parent_child
                    )
                    T_rl_list.append(T_root_child)
                    break
        assert len(T_rl_list) == len(node_traverse_list)

        mesh_list, mesh_color_list = [], []
        for nid, T in zip(node_traverse_list, T_rl_list):
            assert mesh_key in G.nodes[nid].keys()
            mesh_dict = G.nodes[nid][mesh_key].copy()
            mesh = trimesh.Trimesh(
                vertices=np.array(mesh_dict["vertices"]),
                faces=np.array(mesh_dict["faces"])
            )
            mesh.apply_transform(T.copy())
            mesh_list.append(mesh)
        mesh_list = trimesh.util.concatenate(mesh_list)

        MESH.append(mesh_list)
        POSE.append(np.stack(T_rl_list, 0))

    return MESH, POSE


def resolve_range(joint_label, rlim, plim):
    label = np.array(joint_label).argmax()
    if label == 0:  # Screw
        pass
    elif label == 1:  # Revolute
        plim *= 0
    else:  # Prismatic
        rlim *= 0

    return rlim, plim


def sampling_thread(p: tuple) -> None:
    """
    Wrapper function for multiprocessing, calls the sample function.

    Parameters:
    p (tuple): A tuple containing parameters for the sample function.

    Returns:
    None
    """
    pkl_fn, dst_fn, N_states, N_PCL, use_categorical_joint = p
    sample(pkl_fn, dst_fn, N_states, N_PCL, use_categorical_joint)

    return


def sample_point_cloud(
    pcl_source_directory: str, N_states: int, N_pcl: int, use_categorical_joint: bool
) -> None:
    """
    Samples point clouds from graph meshes stored in json files.

    Parameters:
    pcl_source_directory (str): Directory containing the json files.
    N_states (int): Number of states to sample for each graph.
    N_pcl (int): Number of points in each point cloud to sample.

    Returns:
    None
    """
    # Modify the source directory path to create the destination directory
    path_parts = pcl_source_directory.split(os.sep)
    path_parts[path_parts.index("G")] = "PCL"
    pcl_destination_directory = os.sep.join(path_parts)

    # Create the destination directory if it doesn't exist
    try:
        shutil.rmtree(pcl_destination_directory)
    except:
        pass
    os.makedirs(pcl_destination_directory, exist_ok=True)

    # Prepare the list of parameters for multiprocessing
    p_list = []
    for file_name in os.listdir(pcl_source_directory):
        if file_name.endswith(".json"):
            pkl_fn = os.path.join(pcl_source_directory, file_name)
            dst_fn = os.path.join(pcl_destination_directory, file_name[:-5] + ".npz")
            p_list.append((pkl_fn, dst_fn, N_states, N_pcl, use_categorical_joint))

    # Shuffle the list for random processing order
    shuffle(p_list)

    # Process each json file in parallel
    with Pool(os.cpu_count()) as p:
        p.map(sampling_thread, p_list)
