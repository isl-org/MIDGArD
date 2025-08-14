import itertools
import json
from matplotlib import cm
from networkx import Graph
import numpy as np
import networkx as nx
import open3d as o3d
import quaternion
import torch
import trimesh

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, trimesh.Trimesh):
            return {
                "vertices": obj.vertices.tolist(),
                "faces": obj.faces.tolist(),
            }
        return super().default(obj)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis-angle vectors to rotation matrices.

    Args:
        axis_angle (torch.Tensor):
            Rotations represented as axis-angle vectors with shape (..., 3),
            where the magnitude is the rotation angle in radians and the
            direction is the rotation axis.

    Returns:
        torch.Tensor:
            Corresponding rotation matrices with shape (..., 3, 3).
    """
    # Compute the norm of the axis-angle vectors (rotation angles)
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # Shape: (..., 1)
    half_angles = angles * 0.5  # Half angles for quaternion computation

    # Initialize epsilon for numerical stability
    eps = 1e-6
    small_angles = angles.abs() < eps  # Boolean mask for small angles

    # Compute sin(half_angles) / angles, handling small angles with a Taylor approximation
    sin_half_angles_over_angles = torch.empty_like(angles)
    non_small = ~small_angles
    sin_half_angles_over_angles[non_small] = (
        torch.sin(half_angles[non_small]) / angles[non_small]
    )
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] ** 2) / 48.0

    # Construct the quaternion: [cos(half_angle), sin(half_angle)*axis]
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )  # Shape: (..., 4)

    # Normalize the quaternions to ensure valid rotation matrices
    quaternions = quaternions / quaternions.norm(p=2, dim=-1, keepdim=True)

    # Unbind quaternion components
    r, i, j, k = torch.unbind(quaternions, dim=-1)  # Each has shape (...,)

    # Compute intermediate values for the rotation matrix
    two_s = 2.0 / (quaternions * quaternions).sum(-1)  # Shape: (...,)

    # Compute the rotation matrix elements
    ii = i * i
    ij = i * j
    ik = i * k
    ir = i * r
    jj = j * j
    jr = j * r
    jk = j * k
    kk = k * k
    kr = k * r
    rot_elements = torch.stack(
        [
            1 - two_s * (jj + kk),
            two_s * (ij - kr),
            two_s * (ik + jr),
            two_s * (ij + kr),
            1 - two_s * (ii + kk),
            two_s * (jk - ir),
            two_s * (ik - jr),
            two_s * (jk + ir),
            1 - two_s * (ii + jj),
        ],
        dim=-1,
    )  # Shape: (..., 9)

    # Reshape to (..., 3, 3)
    rotation_matrices = rot_elements.reshape(*rot_elements.shape[:-1], 3, 3)

    return rotation_matrices


def _map_obb_to_closest_identity(obb: o3d.geometry.OrientedBoundingBox) -> None:
    """
    Align the given oriented bounding box (OBB) to the canonical axes.
    This function modifies the OBB in place by evaluating all 48 candidates
    (6 axis permutations x 8 sign combinations) and selecting the one that
    maximizes the alignment with the identity matrix.

    Args:
          obb (o3d.geometry.OrientedBoundingBox): The input OBB.
    """
    # Copy original rotation matrix and extents
    R = obb.R.copy()
    extent = obb.extent.copy()
    # Extract the columns of R into a list
    cols = [R[:, 0], R[:, 1], R[:, 2]]
    best_score = -1e9
    best_R = np.empty((3, 3))
    best_extent = np.empty(3)
    # Hard-coded permutations of indices [0, 1, 2]
    permutations = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    # Evaluate all permutations with all sign combinations using itertools.product
    for p in permutations:
        for signs in itertools.product([1, -1], repeat=3):
            c0 = signs[0] * cols[p[0]]
            c1 = signs[1] * cols[p[1]]
            c2 = signs[2] * cols[p[2]]
            # Score candidate: sum of diagonal entries when candidate compared to identity
            score = c0[0] + c1[1] + c2[2]
            if score > best_score:
                best_score = score
                best_R[:, 0] = c0
                best_R[:, 1] = c1
                best_R[:, 2] = c2
                best_extent[0] = extent[p[0]]
                best_extent[1] = extent[p[1]]
                best_extent[2] = extent[p[2]]
    # Update the OBB with the best candidate found
    obb.R = best_R
    obb.extent = best_extent


def compute_minimum_oriented_bounding_box(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.OrientedBoundingBox:
    """
    Compute the minimum oriented bounding box of a mesh.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh.

    Returns:
        tuple: A tuple containing the minimum oriented bounding box center,
               rotation matrix, and size.
    """
    # Compute the minimum oriented bounding box
    obb = mesh.get_minimal_oriented_bounding_box()
    _map_obb_to_closest_identity(obb)
    return obb


def create_bbox_3d(min_coords, max_coords, val=0.5, res=64, as_bool=False):
    """
    Creates a 3D sampled Signed Distance Field (SDF) for a bounding box.

    Args:
        min_coords (iterable): The minimum (x, y, z) coordinates of the bounding box.
        max_coords (iterable): The maximum (x, y, z) coordinates of the bounding box.
        val (float, optional): The value to assign outside the bounding box (or inside if as_bool is False).
                               Defaults to 0.5.
        res (int, optional): The resolution (number of samples along each axis) of the SDF tensor.
                             Defaults to 64.
        as_bool (bool, optional): If True, returns a boolean tensor mask. If False, returns a float tensor.
                                  Defaults to False.

    Returns:
        torch.Tensor: A tensor of shape (res, res, res) representing the sampled SDF or boolean mask.
    """
    if as_bool:
        bbox_3d = torch.ones((res, res, res), dtype=bool)
        val_inside = False
    else:
        bbox_3d = torch.ones((res, res, res), dtype=float) * val
        val_inside = -1 * val
    # assert min_coords.size()[0] == 3 and max_coords.size()[0] == 3
    x_start, y_start, z_start = min_coords[0], min_coords[1], min_coords[2]
    x_end, y_end, z_end = max_coords[0] + 1, max_coords[1] + 1, max_coords[2] + 1
    bbox_3d[x_start:x_end, y_start:y_end, z_start:z_end] = val_inside
    # bbox_3d[z_start:z_end, y_start:y_end, x_start:x_end] = val_inside
    return bbox_3d


def load_normalized_grid_sdf_samples_from_mesh(
    file_path: str,
    grid_resolution: int = 64,
    thresh_clamp_sdf: int = 0.2,
    pad : int = 0.2
) -> tuple:
    """
    Compute a grid-sampled SDF from a mesh.

    Args:
        file_path (str): Path to the mesh file.
        grid_resolution (int): The resolution of the 3D grid used for SDF computation.
        thresh_clamp_sdf (int): The threshold value for clamping the SDF values.
    Returns:
        tuple: A tuple containing the SDF tensor, the extent of the mesh, the minimum and
               maximum voxel coordinates of the bounding box, and the voxel size.
    """
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.is_empty():
        raise ValueError("Loaded mesh is empty or corrupted.")

    # Compute oriented bounding box and rotate mesh to canonical orientation
    min_obb = compute_minimum_oriented_bounding_box(mesh)

    # Compute the transformation matrix for the mesh:
    # For a point p in world coordinates, the transformation is:
    #  p' = R^T * (p - center)
    # Hence, T = [ R^T | -R^T * center ]
    #            [ 0   |       1       ]
    T = np.eye(4)
    T[:3, :3] = min_obb.R.T
    T[:3, 3] = -min_obb.R.T @ min_obb.center

    # Apply transformation to the mesh to put it in cannonical orientation
    mesh = mesh.transform(T)

    # Normalize mesh so its largest dimension equals 1
    aabb = mesh.get_axis_aligned_bounding_box()
    extent = np.array(aabb.get_extent())  # (width, height, depth)
    scale_factor = 1.0 / np.max(extent)
    mesh.scale(scale_factor, center=mesh.get_center())
    extent *= scale_factor

    # SDF generation
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    # Create a grid of coordinates from a bounding box
    bbox_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    bbox_max = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    voxel_size = (bbox_max - bbox_min) / grid_resolution
    bbox_min += 0.5 * voxel_size
    bbox_max -= 0.5 * voxel_size

    # Get the mesh’s axis-aligned bounding box after normalization.
    mesh_aabb = mesh.get_axis_aligned_bounding_box()
    mesh_min = np.asarray(mesh_aabb.get_min_bound())
    mesh_max = np.asarray(mesh_aabb.get_max_bound())

    # The grid is defined over the range [grid_origin, grid_origin + grid_size] where:
    grid_origin = bbox_min  # bbox_min after adjustment (i.e. -0.5 + 0.5 * voxel_size)
    # voxel_size is already computed as (bbox_max - bbox_min) / grid_resolution

    # Map continuous coordinates to discrete grid indices.
    min_coords = np.floor((mesh_min - grid_origin) / voxel_size).astype(np.int32)
    max_coords = np.ceil((mesh_max - grid_origin) / voxel_size).astype(np.int32) - 1

    # Ensure indices are within the valid range [0, grid_resolution - 1]
    min_coords = np.clip(min_coords, 0, grid_resolution - 1)
    max_coords = np.clip(max_coords, 0, grid_resolution - 1)

    # Size [grid_resolution,grid_resolution,grid_resolution]
    coords = [
        np.linspace(bbox_min[i]-pad, bbox_max[i]+pad, num=grid_resolution, dtype=np.float32)
        for i in range(2, -1, -1)
    ]
    grid_components = np.meshgrid(*coords, indexing="ij")
    grid = np.stack(grid_components, axis=-1)
    # grid = np.stack(grid_components[::-1], axis=-1)

    # Compute signed distances and convert to a PyTorch tensor of shape (1, D, D, D)
    signed_distance = scene.compute_signed_distance(grid).numpy()
    sdf = torch.from_numpy(signed_distance).unsqueeze(0)

    # Clamp the SDF values if needed
    if thresh_clamp_sdf != 0.0:
        sdf = torch.clamp(sdf, min=-thresh_clamp_sdf, max=thresh_clamp_sdf)

    return (
        sdf,
        torch.from_numpy(extent),
        min_coords,
        max_coords,
        voxel_size,
    )


def map_upper_triangle_to_list(
    row_index: int, col_index: int, num_nodes_max: int
) -> int:
    """
    Map the row and column indices of an edge in a fully connected, undirected graph to its corresponding position in a linear list of edges.

    In a fully connected graph with `num_nodes_max` nodes, there are `num_nodes_max * (num_nodes_max - 1) / 2` unique edges,
    due to the symmetry of connectivity (i.e., edge from node i to node j is the same as from node j to node i). Storing
    only one instance of each edge significantly reduces memory usage.

    This function calculates the position index in a linear list that represents the upper triangular matrix (without the diagonal)
    of the adjacency matrix of the graph. The upper triangular matrix is chosen because it excludes redundant edges in an undirected graph.

    Args:
        row_index (int): The row index in the upper triangular matrix.
        col_index (int): The column index in the upper triangular matrix, where col_index > row_index.
        num_nodes_max (int): The total number of nodes in the graph.

    Returns:
        int: The index in the linear list where the edge would be stored.

    Note:
    - The list is zero-indexed and the indices are ordered as follows:
      [E_01, E_02, ..., E_0(num_nodes_max-1), E_12, E_13, ..., E_(num_nodes_max-2)(num_nodes_max-1)].
    - The function asserts that row_index < col_index to ensure the indices correspond to the upper triangular matrix.
    """
    assert (
        row_index < col_index
    ), "Error: the provided data structure is not upper triangle"
    return (
        row_index * (2 * num_nodes_max - row_index - 1) // 2 + col_index - row_index - 1
    )


def plucker_need_flip(plucker) -> bool:
    """
    Determine whether the Plücker coordinates need to be flipped or not.

    Args:
        plucker (torch.Tensor): The Plücker coordinates of the edge.

    Returns:
        bool: True if the Plücker coordinates need to be flipped, False otherwise.
    """
    assert (
        plucker.ndim == 1 and plucker.size()[0] == 6
    ), "Plucker coordinates should be a 1D array of size 6."
    x, y, z = plucker[0], plucker[1], plucker[2]

    # Flip if z < 0, or z == 0 and y < 0, or z == 0 and y == 0 and x < 0
    if z < 0 or (z == 0 and (y < 0 or (y == 0 and x < 0))):
        return True
    return False


def load_json(path: str) -> dict:
    """
    Helper function to load a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: The loaded JSON data.
    """
    with open(path, "r") as f:
        return json.load(f)


def write_json(data: dict, filename: str):
    """
    Helper function to write a Python dict to a JSON file.

    Args:
        data (dict): The Python dictionary to be written.
        path (str): Path to the JSON file.
    """
    with open(filename, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)


def preprocess_and_pack(
    node_features_dict: list,
    edge_features_dict: list,
    num_nodes_max: int = 25,
    permute: bool = True,
    fixed_chirality: bool = False,
    use_categorical_joint=False,
    use_node_orientation=False,
    use_manifold_plucker=False,
):
    """
    Processes and packs node and edge features of a graph into a compact format suitable for input into neural network models.

    The function assumes a fully connected, non-oriented graph structure and operates under the assumption that for a graph with a maximum of `num_nodes_max` nodes, there are `num_nodes_max * (num_nodes_max - 1) / 2` possible edges.

    Args:
        node_features_dict (list): A list of dictionaries containing node features.
        edge_features_dict (list): A list of dictionaries containing edge features.
        num_nodes_max (int): The maximum number of nodes in the graph.
        permute (bool): Whether to randomly permute the nodes.
        fixed_chirality (bool): Whether to fix the chirality of the edges.
        use_categorical_joint (bool): Whether to use categorical joint labels.
        use_node_orientation (bool): Whether to use node orientation features.
        use_manifold_plucker (bool): Whether to use the manifold Plücker coordinates.

    Returns:
        tuple: A tuple containing the processed node features, edge features, and node mapping.
    """

    ###############################
    ### Node Feature Processing ###
    ###############################

    num_nodes = len(node_features_dict)
    if num_nodes > num_nodes_max:
        print(f"Warning, extend {num_nodes_max} to {num_nodes}")
        num_nodes_max = num_nodes
    num_empty = num_nodes_max - num_nodes

    # In the origin index, the first num_nodes are object
    node_mask = np.zeros(num_nodes_max, dtype=bool)
    node_mask[:num_nodes] = True
    # Randomly shuffle nodes?
    if permute:
        node_map = np.random.permutation(
            num_nodes_max
        ).tolist()  # stores the original id
        node_mask = [node_mask[i] for i in node_map]  # Adjust the node mask
    else:
        node_map = np.arange(num_nodes_max).tolist()

    # Get the raw node features
    if use_node_orientation:
        raw_node_features_bbox = [
            np.array(node_features["obb_size"].split(), dtype=np.double)
            for node_features in node_features_dict
        ] + [np.zeros(3)] * num_empty
        raw_node_features_t_gl = [
            (
                node_features["abs_center"]
                + np.array(node_features["obb_center"].split(), dtype=np.double)
            )
            for node_features in node_features_dict
        ] + [np.zeros(3)] * num_empty
        raw_obb_t_gl = [
            np.array(node_features["obb_center"].split(), dtype=np.double)
            for node_features in node_features_dict
        ] + [np.zeros(3)] * num_empty
        raw_node_features_q_gl = [
            quaternion.from_float_array(
                np.array(node_features["obb_quat"].split(), dtype=np.double)
            )
            for node_features in node_features_dict
        ] + [quaternion.one] * num_empty
        raw_node_features_r_gl = [
            quaternion.as_rotation_vector(quat) for quat in raw_node_features_q_gl
        ]
    else:
        raw_node_features_bbox = [
            node_features["bbox_L"] for node_features in node_features_dict
        ] + [np.zeros(3)] * num_empty
        raw_node_features_t_gl = [
            node_features["abs_center"] for node_features in node_features_dict
        ] + [np.zeros(3)] * num_empty
        raw_obb_t_gl = np.zeros_like(raw_node_features_t_gl)
        raw_node_features_r_gl = np.zeros_like(raw_node_features_t_gl)

    # Rearrange features to match the potentially shuffled nodes order
    node_features_bbox = [raw_node_features_bbox[i] for i in node_map]
    node_features_t_gl = [raw_node_features_t_gl[i] for i in node_map]
    node_features_r_gl = [raw_node_features_r_gl[i] for i in node_map]
    obb_t_gl = [raw_obb_t_gl[i] for i in node_map]

    # Bounding box node feature
    node_features_bbox = torch.from_numpy(np.stack(node_features_bbox, axis=0)).float()

    # Transform node feature: p_global = np.matmul(T_gl, p_local_)
    node_features_t_gl = torch.from_numpy(np.stack(node_features_t_gl, axis=0)).float()

    # In the NAP paper, it is assumed that R = I; here we do not make this assumption...
    node_features_r_gl = torch.from_numpy(np.stack(node_features_r_gl, axis=0)).float()

    # Transform node feature: p_global = np.matmul(T_gl, p_local_)
    obb_t_gl = torch.from_numpy(np.stack(obb_t_gl, axis=0)).float()

    # Build the node feature list following the NAP paper convention
    # in NAP node_features: [mask_occ(1), bbox(3), r_gl(3), t_gl(3) | Object Neural Code]
    node_features_data = torch.cat(
        [
            torch.LongTensor(node_mask)[..., None],
            node_features_bbox,
            node_features_r_gl,  # Axis-angle representation, where the magnitude encodes the angle.
            node_features_t_gl,
        ],
        -1,
    )

    ###############################
    ### Edge Feature Processing ###
    ###############################

    total_edges = int(num_nodes_max * (num_nodes_max - 1) / 2)  # Include invalid
    if use_categorical_joint:
        edge_features_label = torch.zeros(
            (total_edges), dtype=torch.long
        )  # Joint labels

    edge_features_lim = torch.zeros(
        (total_edges, 4), dtype=torch.float32
    )  # Joint limits

    if use_manifold_plucker:
        edge_features_plucker = torch.zeros((total_edges, 5), dtype=torch.float32)
    else:
        edge_features_plucker = torch.zeros(
            (total_edges, 6), dtype=torch.float32
        )  # Plucker parametrization of the edges

    edge_features_chirality = torch.zeros((total_edges), dtype=torch.long)  # Chirality

    # Following NAP convention, by default, the list of edges represent the upper triangle, i.e. row i, col j, then i < j
    for edge_features in edge_features_dict:
        # Get the raw edge features
        raw_edge_features_src_ind = edge_features["e0"]["src_ind"]
        raw_edge_features_dst_ind = edge_features["e0"]["dst_ind"]
        plucker = edge_features["e0"]["plucker"]

        # Rearrange features to match the potentially shuffled nodes order
        edge_features_src_ind = node_map.index(raw_edge_features_src_ind)
        edge_features_dst_ind = node_map.index(raw_edge_features_dst_ind)
        edge_features_r_gl = node_features_r_gl[edge_features_src_ind]
        edge_features_t_gl = node_features_t_gl[edge_features_src_ind]

        # Transform the Plücker coordinates from local to global frame
        R_obb = axis_angle_to_matrix(edge_features_r_gl).t()
        t_obb = -obb_t_gl[edge_features_src_ind]
        plucker_global = torch.from_numpy(plucker.copy()).float()
        edge_features_R_gl = axis_angle_to_matrix(edge_features_r_gl)
        edge_features_lg = R_obb @ edge_features_R_gl @ plucker_global[:3]
        edge_features_mg = R_obb @ edge_features_R_gl @ plucker_global[
            3:
        ] + torch.linalg.cross(edge_features_t_gl + t_obb, edge_features_lg)
        plucker_global = torch.cat([edge_features_lg, edge_features_mg], 0)

        # Orient the global plucker to hemisphere
        flip = plucker_need_flip(plucker_global)
        if flip:
            plucker_global = -plucker_global

        if edge_features_src_ind > edge_features_dst_ind:  # i = dst, j = src
            i, j = edge_features_dst_ind, edge_features_src_ind
            flip = (
                not flip
            )  # when reverse the src and dst, the plucker parametrization should be multiplied by -1.0
        elif edge_features_src_ind < edge_features_dst_ind:
            i, j = edge_features_src_ind, edge_features_dst_ind
        else:
            raise ValueError("edge_features_src_ind == edge_features_dst_ind")

        # Organize the edge features into an ordered list E_01, E_02, ..., E_12, E_23,...
        edge_list_ind = map_upper_triangle_to_list(i, j, num_nodes_max)

        # Resolve edge chirality
        if fixed_chirality:
            if flip:  # Force not-fliped!
                plucker_global = -plucker_global
            edge_features_chirality[edge_list_ind] = 1
        else:
            if flip:  # 2 is flip plucker
                edge_features_chirality[edge_list_ind] = 2
            else:  # 1 is not flip plucker
                edge_features_chirality[edge_list_ind] = 1

        # edge_features_lim[edge_list_ind, :2] = torch.Tensor(edge_features["r_limits"])
        # edge_features_lim[edge_list_ind, 2:] = torch.Tensor(edge_features["p_limits"])
        if use_manifold_plucker:
            edge_features_plucker[edge_list_ind] = plucker_to_manifold(plucker_global)
        else:
            edge_features_plucker[edge_list_ind] = plucker_global

        if use_categorical_joint:
            label, r_limits, p_limits = _resolve_edge_features(
                edge_features["r_limits"], edge_features["p_limits"]
            )
            edge_features_label[edge_list_ind] = torch.Tensor(label)
            edge_features_lim[edge_list_ind, :2] = torch.Tensor(r_limits)
            edge_features_lim[edge_list_ind, 2:] = torch.Tensor(p_limits)
        else:
            edge_features_lim[edge_list_ind, :2] = torch.Tensor(
                edge_features["r_limits"]
            )
            edge_features_lim[edge_list_ind, 2:] = torch.Tensor(
                edge_features["p_limits"]
            )

    # Turn the 0,1,2 chirality value into 3D normalized codes [0,0,1], [0,1,0], [1,0,0]
    edge_features_chirality = torch.nn.functional.one_hot(
        edge_features_chirality, num_classes=3
    ).float()

    if use_categorical_joint:
        # Turn the 0,1,2 edge_features_label value into 3D normalized codes [0,0,1], [0,1,0], [1,0,0]
        edge_features_label = torch.nn.functional.one_hot(
            edge_features_label, num_classes=3
        ).float()

        # Build the edge feature data using categorical data
        edge_features_data = torch.cat(
            [
                edge_features_chirality,
                edge_features_label,
                edge_features_plucker,
                edge_features_lim,
            ],
            dim=1,
        )
    else:
        # Build the edge feature data following the NAP paper convention
        #
        # if use_manifold_plucker: edge_features: [chirality(3), plucker(6), rlim(2), plim(2)]
        # else: edge_features: [chirality(3), plucker(4), rlim(2), plim(2)]
        edge_features_data = torch.cat(
            [edge_features_chirality, edge_features_plucker, edge_features_lim], dim=1
        )

    return node_features_data, edge_features_data, node_map


def plucker_to_manifold(plucker) -> torch.Tensor:
    """
    Convert Plücker coordinates from the standard parametrization to the manifold parametrization.

    Args:
        plucker (torch.Tensor): The Plücker coordinates of the edge.

    Returns:
        torch.Tensor: The Plücker coordinates in the manifold parametrization.
    """
    l, m = plucker[..., 0:3], plucker[..., 3:6]

    # Ensure the vector is normalized before computing angles
    magnitude = torch.norm(l)

    assert torch.abs(magnitude - 1.0) < 1e-4, "The vector is not normalized"

    # Use spherical coordinate parametrization to ensure normalization
    theta = torch.acos(torch.clamp(l[2], min=-1, max=1))  # Range of theta is 0 to pi
    phi = torch.atan2(l[1], l[0])  # Range of phi is -pi to pi
    if phi < 0:
        phi += 2 * torch.pi  # Normalizing range to 0 to 2*pi

    # Find a basis vector that is not collinear to l by checking if the cross product is non-zero
    n = torch.linalg.cross(l, m)

    plucker_manifold = torch.cat((torch.tensor([phi, theta]), n), axis=0)

    return plucker_manifold


def manifold_to_plucker(plucker_manifold):
    """
    Convert Plücker coordinates from the manifold parametrization to the standard Plücker parametrization.

    Args:
        plucker_manifold: [phi, theta, n]

    Returns:
        plucker: [l, m]
    """
    phi = plucker_manifold[0]
    theta = plucker_manifold[1]
    n = plucker_manifold[2:]
    l = np.array(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
    )
    m = np.cross(n, l)

    return np.concatenate((l, m), axis=0)


def _resolve_edge_features(r_lim, p_lim, p_range_th=1e-3, r_range_th=1e-3):
    """
    Resolve the edge features based on the limits.

    Args:
        r_lim: The rotation limits.
        p_lim: The prismatic limits.
        p_range_th: The threshold for the prismatic range.
        r_range_th: The threshold for the rotation range.

    Returns:
        edge_features_label: The resolved edge features label.
        r_lim: The resolved rotation limits.
        p_lim: The resolved prismatic limits.
    """
    if r_lim[1] < r_lim[0]:
        r_lim[0], r_lim[1] = r_lim[1], r_lim[0]

    if p_lim[1] < p_lim[0]:
        p_lim[0], p_lim[1] = p_lim[1], p_lim[0]

    r_range = abs(r_lim[1] - r_lim[0])
    p_range = abs(p_lim[1] - p_lim[0])
    if r_range > r_range_th and p_range > p_range_th:  # Screw
        edge_features_label = [0]
    elif r_range > r_range_th:
        edge_features_label = [1]
        p_lim = [0.0 for _ in p_lim]
    else:  # Prismatic
        edge_features_label = [2]
        r_lim = [0.0 for _ in r_lim]

    return edge_features_label, r_lim, p_lim


def uppertri_list_to_matrix(trilist: torch.Tensor) -> torch.Tensor:
    """
    Converts a flattened upper triangular matrix (as a vector) back into a 2D square matrix.

    This function is designed to reconstruct a full 2D matrix from its upper triangular
    elements, excluding the diagonal. It is particularly useful when dealing with symmetric
    matrices where the upper triangle contains all the unique information, such as in the case
    of adjacency matrices or correlation matrices in graphs.

    Args:
        trilist (torch.Tensor): A 1-dimensional tensor containing the upper triangular elements
        of a square matrix. Its length should be n(n-1)/2, where n is the size of the resulting
        square matrix.

    Returns:
        matrix (torch.Tensor): A 2D square matrix of size n x n, where the upper triangular
        elements are filled with the values from `trilist`, and the lower triangular and the
        diagonal elements are filled with zeros.
    """
    # Determine the size of the square matrix based on the length of the input list
    N = len(trilist)
    num_nodes_max = int(np.ceil(np.sqrt(2 * N)))

    # Create a mask for the upper triangular indices (excluding the diagonal)
    r = torch.arange(num_nodes_max)
    mask = r[:, None] < r

    # Initialize a square matrix with zeros
    matrix = torch.zeros(
        num_nodes_max, num_nodes_max, trilist.shape[1], dtype=trilist.dtype
    )

    # Fill the upper triangular part of the matrix with the input list
    matrix[mask] = trilist

    return matrix


def get_G_from_VE(
    node_features_dict,
    edge_features_dict,
    configuration=None,
    feature_size=None,
    feature_index=None,
    body_categories=None,
    asset_categories=None,
) -> Graph:
    """
    Construct a graph from node and edge feature arrays using NetworkX.

    The function assumes that the input tensors are in the CPU memory and converts them to numpy arrays if necessary.
    It then constructs a graph where nodes represent objects with attributes like bounding boxes, rotation,
    and translation matrices, while edges represent relationships between these objects with attributes like
    chirality, Plücker coordinates, and limits. Additional node features are added if available.
    The graph is built by first adding nodes with their attributes and then connecting the nodes with edges,
    setting edge attributes based on the features provided. The transformation matrices for the edges are
    computed within the graph space.

    Args:
        node_features_dict (torch.Tensor): A tensor containing node features.
        edge_features_dict (torch.Tensor): A tensor containing edge features.
        configuration (dict): A dictionary containing the configuration parameters.
        feature_size (dict): A dictionary containing the size of the node and edge features.
        feature_index (dict): A dictionary containing the index of the node and edge features.
        body_categories (list): A list of body categories.
        asset_categories (list): A list of asset categories.

    Returns:
        G: A NetworkX Graph object with nodes and edges populated with their respective attributes.
    """

    if isinstance(node_features_dict, torch.Tensor):
        node_features_dict = node_features_dict.cpu().numpy()
    if isinstance(edge_features_dict, torch.Tensor):
        edge_features_dict = edge_features_dict.cpu().numpy()

    node_mask = node_features_dict[:, 0] > 0.5
    num_nodes_max = len(node_mask)
    assert len(edge_features_dict) == int(
        num_nodes_max * (num_nodes_max - 1) / 2
    ), f"len(edge_features_dict)={len(edge_features_dict)}, num_nodes_max={num_nodes_max}"

    node_features = node_features_dict[node_mask]
    num_nodes = len(node_features)

    # Create the networkx graph object
    G = nx.Graph()

    if num_nodes >= 2:
        # Fill in NODE attributes
        idx_sa = feature_index["node_sa"]
        idx_sb = feature_index["node_sb"]
        idx_bb = feature_index["node_bb"]
        idx_r = feature_index["node_r"]
        # because feature_index["node_t"] = feature_index["node_r"]
        # when rotation is not activated:
        idx_t = idx_r + feature_size["node_r"]
        idx_2d = idx_t + feature_size["node_t"]
        idx_c = feature_index["edge_c"]
        idx_j = feature_index["edge_j"]
        idx_pm = feature_index["edge_p"]
        idx_l = feature_index["edge_l"]

        node_color_list = cm.tab20c(np.linspace(0, 1, num_nodes + 1))[:-1]

        for node_id in range(num_nodes):
            G.add_node(node_id)

            _r = torch.as_tensor(node_features[node_id][idx_r:idx_t])
            R = axis_angle_to_matrix(_r).cpu().numpy()

            node_attr = {
                "node_id": node_id,
                "bbox": node_features[node_id][idx_bb:idx_r],
                "R": R.copy(),
                "t": node_features[node_id][idx_t : idx_t + 3].copy(),
                "color": node_color_list[node_id],
                "mesh_RT": R.T,
            }

            if configuration["use_categorical_asset"] and not (
                asset_categories == None
            ):
                assert len(asset_categories) == idx_sb - idx_sa
                if torch.as_tensor(node_features[node_id][idx_sa:idx_sb]).any():
                    node_attr["asset_id"] = asset_categories[
                        torch.argmax(
                            torch.as_tensor(node_features[node_id][idx_sa:idx_sb])
                        )
                    ]
                else:
                    node_attr["asset_id"] = "Unknown"

            if configuration["use_categorical_body"] and not (body_categories == None):
                assert len(body_categories) == idx_bb - idx_sb
                if torch.as_tensor(node_features[node_id][idx_sb:idx_bb]).any():
                    node_attr["body_id"] = body_categories[
                        torch.argmax(
                            torch.as_tensor(node_features[node_id][idx_sb:idx_bb])
                        )
                    ]
                else:
                    node_attr["body_id"] = "Empty"

            if (
                configuration["use_categorical_asset"]
                or configuration["use_categorical_body"]
            ):
                node_attr["categorical"] = node_features[node_id][idx_sa:idx_bb]

            if configuration["use_node_2D_preencoded_latent"]:
                node_attr["2d_latent"] = node_features[node_id][idx_2d:]

            nx.set_node_attributes(G, {node_id: node_attr})

        # Fill in EDGE attributes
        original_node_id = np.arange(num_nodes_max)[node_mask].tolist()
        for _i in range(num_nodes_max):
            for _j in range(num_nodes_max):
                if _i >= _j:
                    continue
                # src = i, dst = j
                ind = map_upper_triangle_to_list(_i, _j, num_nodes_max)
                edge_features_chirality = edge_features_dict[ind, idx_c:idx_j].argmax()
                if edge_features_chirality == 0:
                    continue
                assert (
                    _i in original_node_id and _j in original_node_id
                ), "invalid edge detected!"
                src_i, dst_j = original_node_id.index(_i), original_node_id.index(_j)

                G.add_edge(src_i, dst_j)
                T_gi, T_gj = np.eye(4), np.eye(4)
                T_gi[:3, :3] = G.nodes[src_i]["R"]  # Node i wrt global
                T_gi[:3, 3] = G.nodes[src_i]["t"]  # Node i wrt global
                T_gj[:3, :3] = G.nodes[dst_j]["R"]  # Node j wrt global
                T_gj[:3, 3] = G.nodes[dst_j]["t"]  # Node j wrt global
                T_ig = np.linalg.inv(T_gi).copy()  # Global wrt node j
                T_ij = T_ig.copy() @ T_gj.copy()  # T0 = node j wrt node i

                # Plucker
                if configuration["use_manifold_plucker"]:
                    edge_features_plucker = manifold_to_plucker(
                        edge_features_dict[ind, idx_pm:idx_l]
                    )
                else:
                    edge_features_plucker = edge_features_dict[ind, idx_pm:idx_l]

                # Flip Plucker axis based on chirality
                if edge_features_chirality == 2:
                    edge_features_plucker = -edge_features_plucker

                local_plucker = edge_features_plucker.copy()
                li = T_ig[:3, :3] @ local_plucker[:3]
                mi = T_ig[:3, :3] @ local_plucker[3:] + np.cross(T_ig[:3, 3], li)
                local_plucker = np.concatenate([li, mi])

                # Joint limits
                e_rlim = edge_features_dict[ind, idx_l : idx_l + 2]
                e_plim = edge_features_dict[ind, idx_l + 2 : idx_l + 4]

                edge_attributes = {
                    (src_i, dst_j): {
                        "src": src_i,
                        "dst": dst_j,
                        "T_src_dst": T_ij.copy(),
                        "plucker": local_plucker.copy(),
                        "plim": e_plim.copy(),
                        "rlim": e_rlim.copy(),
                        # additional info
                        "global_plucker": edge_features_plucker.copy(),  # for computing parameter space distance
                        "T_ig": T_ig.copy(),
                        "T_gj": T_gj.copy(),
                    }
                }
                if configuration["use_categorical_joint"]:
                    e_label = edge_features_dict[ind, idx_j:idx_pm]
                    edge_attributes[(src_i, dst_j)]["joint_label"] = e_label.copy()

                nx.set_edge_attributes(G, edge_attributes)

    return G
