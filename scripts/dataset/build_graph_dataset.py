import argparse
from joblib import Parallel, delayed
import json
import logging
import numpy as np
import os
import open3d as o3d
import quaternion
import re
import trimesh
import subprocess
import sys
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from scripts.utils.urdfpy.urdfpy import URDF, configure_origin
from core.utils.mujoco_utils import (
    generate_mujoco_scene_from_VE,
)
from core.utils.data_utils import compute_minimum_oriented_bounding_box

# Configure the logger (do this once in the main entry point of your application)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class midgard_JointLimit(object):
    """The limits of the joint (code taken from URDFPY).

    Parameters
    ----------
    effort : float, optional
        The maximum joint effort (N for prismatic joints, Nm for revolute).
    velocity : float, optional
        The maximum joint velocity (m/s for prismatic joints, rad/s for
        revolute).
    lower : float, optional
        The lower joint limit (m for prismatic joints, rad for revolute).
    upper : float, optional
        The upper joint limit (m for prismatic joints, rad for revolute).
    """

    def __init__(self, lower_lin=None, upper_lin=None, lower_ang=None, upper_ang=None):
        self.lower_lin = lower_lin
        self.upper_lin = upper_lin
        self.lower_ang = lower_ang
        self.upper_ang = upper_ang

    @property
    def lower_lin(self):
        """float : The lower joint limit."""
        return self._lower_lin

    @lower_lin.setter
    def lower_lin(self, value) -> None:
        if value is not None:
            value = float(value)
        self._lower_lin = value

    @property
    def upper_lin(self):
        """float : The upper joint limit."""
        return self._upper_lin

    @upper_lin.setter
    def upper_lin(self, value) -> None:
        if value is not None:
            value = float(value)
        self._upper_lin = value

    @property
    def lower_ang(self):
        """float : The lower joint limit."""
        return self._lower_ang

    @lower_ang.setter
    def lower_ang(self, value) -> None:
        if value is not None:
            value = float(value)
        self._lower_ang = value

    @property
    def upper_ang(self):
        """float : The upper joint limit."""
        return self._upper_ang

    @upper_ang.setter
    def upper_ang(self, value) -> None:
        if value is not None:
            value = float(value)
        self._upper_ang = value


class midgard_Joint(object):
    """Joint data container (code taken from URDFPY)"""

    TYPES = ["prismatic", "revolute", "continuous", "screw"]

    def __init__(
        self,
        name,
        joint_type,
        parent,
        child,
        origin=None,
        axis=None,
        limit=None,
        direct=None,
    ) -> None:
        self.name = name
        self.joint_type = joint_type
        self.parent = parent
        self.child = child
        self.origin = origin
        self.axis = axis
        self.limit = limit
        self.direct = direct

    @property
    def name(self) -> str:
        """str : Name for this joint."""
        return self._name

    @name.setter
    def name(self, value) -> None:
        self._name = str(value)

    @property
    def joint_type(self) -> str:
        """str : The type of this joint."""
        return self._joint_type

    @joint_type.setter
    def joint_type(self, value) -> None:
        value = str(value)
        if value not in midgard_Joint.TYPES:
            raise ValueError("Unsupported joint type {}".format(value))
        self._joint_type = value

    @property
    def parent(self) -> str:
        """str : The name of the parent link."""
        return self._parent

    @parent.setter
    def parent(self, value) -> None:
        self._parent = str(value)

    @property
    def child(self) -> str:
        """str : The name of the child link."""
        return self._child

    @child.setter
    def child(self, value) -> None:
        self._child = str(value)

    @property
    def axis(self):
        """(3,) float : The joint axis in the joint frame."""
        return self._axis

    @axis.setter
    def axis(self, value) -> None:
        if value is None:
            value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif np.linalg.norm(value) < 1e-4:
            value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            value = np.asanyarray(value, dtype=np.float32)
            if value.shape != (3,):
                raise ValueError("Invalid shape for axis, should be (3,)")
            value = value / np.linalg.norm(value)
        self._axis = value

    @property
    def origin(self):
        """(4,4) float : The pose of child and joint frames relative to the
        parent link's frame.
        """
        return self._origin

    @origin.setter
    def origin(self, value) -> None:
        self._origin = configure_origin(value)

    @property
    def limit(self):
        """:class:`.JointLimit` : The limits for this joint."""
        return self._limit

    @limit.setter
    def limit(self, value) -> None:
        if value is None:
            if self.joint_type in ["prismatic", "revolute", "screw"]:
                raise ValueError(
                    "Require joint limit for prismatic, revolute and screw joints"
                )
        elif not isinstance(value, midgard_JointLimit):
            raise TypeError("Expected JointLimit type")
        self._limit = value

    @property
    def direct(self) -> bool:
        """str : Direct or inverse joint."""
        return self._direct

    @direct.setter
    def direct(self, value) -> None:
        self._direct = bool(value)


def load_urdf(file: str, show: bool = False, animate: bool = False):
    """
    Load specified URDF file into a data structure. Optionally, show or animate the robot.

    Parameters:
        - file (str): Path to the URDF file.
        - show (bool): If True, display the robot after loading.
        - animate (bool): If True, animate the robot after loading.

    Returns:
        tuple: (robot, fk, status) where `robot` is the robot model, `fk` is forward kinematics, and
        `status` is a boolean indicating success.
    """
    try:
        # Parse the URDF file
        robot = URDF.load(file)

        # Compute FK
        fk = robot.link_fk()

        if show:  # Show the robot
            robot.show()
        elif animate:  # Animate the robot
            robot.animate()

        return robot, fk, True

    except Exception as e:
        logger.error(f"Unexpected error: {e} while loading {file}")
        return None, None, False


def check_graph_consistency_with_nap(
    path_to_midgard_graph,
    path_to_nap_graph,
    code,
) -> None:
    """
    Checks for consistency between two graph representations (NAP and Midgard) by comparing
    node and edge properties.

    The function loads two graph files (in .npz format) and compares their node and edge arrays.
    It verifies that the number of nodes/edges match and then iterates over nodes and edges to
    compare specific fields using a defined relative and absolute tolerance. Any discrepancies are
    logged via the logging module.

    Args:
        path_to_midgard_graph (str): File path to the Midgard graph (.npz format).
        path_to_nap_graph (str): File path to the NAP graph (.npz format).
        code (str): An identifier code used for logging error messages.

    Returns:
        None
    """
    # Define tolerance
    rtol = 1e-3  # Relative tolerance
    atol = 1e-5  # Absolute tolerance
    errors = []

    try:
        data_nap = np.load(path_to_nap_graph, allow_pickle=True)
        data_midgard = np.load(path_to_midgard_graph, allow_pickle=True)
    except:
        logging.error(f"ERROR: unable to load graph {code}")
        return

    try:
        # Check nodes
        if data_nap["V"].shape != data_midgard["V"].shape:
            errors.append(f"{code}: mismatch in the number of nodes")
        if data_nap["E"].shape != data_midgard["E"].shape:
            errors.append(f"{code}: mismatch in the number of edges")

        # Node details
        for node_idx in range(len(data_nap["V"])):
            node_nap = data_nap["V"][node_idx]
            node_midgard = data_midgard["V"][node_idx]

            if node_nap["names"] != node_midgard["names"]:
                errors.append(f"{code}: mismatch in node names at index {node_idx}")

            for field in ["mesh_T_lm_list", "abs_center", "bbox_L"]:
                if not np.allclose(
                    np.array(node_nap[field]),
                    np.array(node_midgard[field]),
                    rtol=rtol,
                    atol=atol,
                ):
                    errors.append(
                        f"{code}: mismatch in {field} at node index {node_idx}"
                    )

        # Edge details
        for edge_idx in range(len(data_nap["E"])):
            edge_nap = data_nap["E"][edge_idx]
            edge_midgard = data_midgard["E"][edge_idx]
            if edge_nap["name"] != edge_midgard["name"]:
                errors.append(
                    f"{code}: mismatch in edge names at index {node_idx}: {edge_nap['name']} vs {edge_midgard['name']}"
                )

            for sub_field in ["r_limits", "p_limits", "e0", "e1"]:
                if sub_field in ["e0", "e1"]:
                    # Detailed checks within sub-fields of edges
                    for detail in ["plucker", "axis", "T0", "T_src_j1", "T_j2_dst"]:
                        if not np.allclose(
                            np.array(edge_nap[sub_field][detail]),
                            np.array(edge_midgard[sub_field][detail]),
                            rtol=rtol,
                            atol=atol,
                        ):
                            errors.append(
                                f"{code}: mismatch in {sub_field} {detail} at edge index {edge_idx}: {edge_nap[sub_field][detail]} vs {edge_midgard[sub_field][detail]}"
                            )
                    for str_detail in ["src_name", "dst_name", "src_ind", "dst_ind"]:
                        if (
                            edge_nap[sub_field][str_detail]
                            != edge_midgard[sub_field][str_detail]
                        ):
                            errors.append(
                                f"{code}: mismatch in {sub_field} {str_detail} at edge index {edge_idx}: {edge_nap[sub_field][str_detail]} vs {edge_midgard[sub_field][str_detail]}"
                            )
                else:
                    if not np.allclose(
                        np.array(edge_nap[sub_field]),
                        np.array(edge_midgard[sub_field]),
                        rtol=rtol,
                        atol=atol,
                    ):
                        errors.append(
                            f"{code}: mismatch in {sub_field} at edge index {edge_idx}: {edge_nap[sub_field]} vs {edge_midgard[sub_field]}"
                        )

        if errors:
            for error in errors:
                logging.error(error)
        else:
            print(f"{code}: OK")
    except Exception as e:
        print(f"{code}: ERROR! -- {e}")
        return


def export_off(vertices, triangles, filename) -> None:
    """
    Exports a mesh in the OFF file format.

    https://github.com/JiahuiLei/NAP/blob/main/core/models/utils/occnet_utils/utils/libmcubes/exporter.py

    Args:
        vertices (iterable): An array-like structure containing vertex coordinates (each a sequence of 3 floats).
        triangles (iterable): An array-like structure containing triangle indices (each a sequence of 3 integers).
        filename (str): The path to the output file where the mesh will be saved.

    Returns:
        None
    """

    with open(filename, "w") as fh:
        fh.write("OFF\n")
        fh.write("{} {} 0\n".format(len(vertices), len(triangles)))

        for v in vertices:
            fh.write("{:.10f} {:.10f} {:.10f}\n".format(*v))

        for f in triangles:
            fh.write("3 {} {} {}\n".format(*f))


def recenter_aggregate_mesh(link_object, fix_mesh=False):
    """
    Recenters and aggregates the mesh of a link.

    This function takes a link object that contains one or more visual meshes. It aggregates the meshes
    by concatenating their vertices and faces, computes a translation needed to center the combined mesh,
    and returns the transformation and aggregated mesh. Optionally, it attempts to repair the mesh.

    Args:
        link_object: An object representing a robot link. It must have a 'visuals' attribute, where each
                     visual contains a geometry mesh with 'vertices' and 'faces'.
        fix_mesh (bool, optional): If True, attempts to repair the aggregated mesh (fix inversion, normals,
                                   winding, and fill holes). Defaults to False.

    Returns:
        tuple: A tuple containing:
            - abs_center (np.ndarray): The absolute center of the aggregated (and re-centered) mesh.
            - bbox_L (np.ndarray): The half-extent (bounding box lengths) of the aggregated mesh.
            - mesh_T_lm_list (np.ndarray): A list of 4x4 transformation matrices (one per visual) to recenter the mesh.
            - agg_trimesh (trimesh.Trimesh): The aggregated and processed mesh.
    """
    meshes_list = [visual.geometry.mesh.meshes for visual in link_object.visuals]
    meshes = [item for sublist in meshes_list for item in sublist]
    vertices = np.concatenate(
        [np.asarray(mesh.vertices) for mesh in meshes],
        axis=0,
    )
    faces = np.concatenate([np.asarray(mesh.faces) for mesh in meshes], axis=0)

    assert vertices.shape[0] > 0, "The provided vector of vertices should not be empty!"
    assert (
        vertices.shape[1] == 3
    ), "The provided vector of vertices should be of size (N, 3), where N is the number of vertices."

    # Compute the coordinate shift needed to center the mesh
    t = -0.5 * (vertices.max(axis=0) + vertices.min(axis=0))
    recenter_transform = np.identity(4)
    recenter_transform[0:3, 3] = t
    mesh_T_lm_list = np.array(
        [recenter_transform.copy() for _ in range(len(link_object.visuals))]
    )
    abs_center = -t

    # Since we've concatenated the vertices, we need to update the face indices
    face_offset = 0
    new_faces = []
    for mesh in meshes:
        new_faces.append(mesh.faces + face_offset)
        face_offset += len(mesh.vertices)
    faces = np.concatenate(new_faces, axis=0)

    # Create a new mesh from all -- centered -- vertices and faces
    agg_trimesh = trimesh.Trimesh(vertices=vertices + t, faces=faces)

    # Resolve duplicates and make the mesh watertight if possible
    agg_trimesh.merge_vertices()  # This also removes duplicate vertices

    # Clean up the mesh
    if fix_mesh:
        trimesh.repair.fix_inversion(agg_trimesh)
        trimesh.repair.fix_normals(agg_trimesh)
        trimesh.repair.fix_winding(agg_trimesh)
        trimesh.repair.fill_holes(agg_trimesh)

    # Compute the bounding box
    bbox_L = 0.5 * (agg_trimesh.vertices.max(axis=0) - agg_trimesh.vertices.min(axis=0))
    bbox_L = np.clip(bbox_L, 1e-3, None)  # Ensures that values are at least 1e-2

    return abs_center, bbox_L, mesh_T_lm_list, agg_trimesh


def sample_mesh(tmesh, number_point_samples):
    """
    Samples points on the surface of a mesh.

    The function uses trimesh's sampling routines to obtain a specified number of points
    from the mesh surface. If the initial sampling does not yield enough points, additional
    samples are concatenated until the target number is reached.

    Args:
        tmesh (trimesh.Trimesh): The input mesh from which to sample points.
        number_point_samples (int): The number of sample points desired.

    Returns:
        np.ndarray: An array of shape (number_point_samples, 3) containing the sampled points,
                    stored as type np.float16.
    """
    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.CRITICAL)
    pcl, _ = trimesh.sample.sample_surface_even(tmesh, number_point_samples * 2)
    while len(pcl) < number_point_samples:
        _pcl, _ = trimesh.sample.sample_surface_even(tmesh, number_point_samples * 2)
        pcl = np.concatenate([pcl, _pcl])
    pcl = pcl[:number_point_samples]
    pcl = np.asarray(pcl, dtype=np.float16)
    return pcl


def compute_plucker_coordinates(axis, joint_transform):
    """
    Computes the Plücker coordinates for a joint.

    Given a joint axis and a transformation matrix, this function calculates the unit direction
    vector (l) and moment vector (m) that together form the Plücker coordinates for the joint.

    Args:
        axis (np.ndarray): A 3D vector representing the joint axis.
        joint_transform (np.ndarray): A 4x4 transformation matrix representing the joint pose.

    Returns:
        np.ndarray: A concatenated array of shape (6,) where the first three elements are the unit direction vector (l)
                    and the next three are the moment vector (m).
    """
    # Get the unit direction vector (l) in the global frame
    R = joint_transform[0:3, 0:3]
    l = np.dot(R.T, axis)

    # Compute moment vector (m) as A x B, where A and B are points on the line.
    # Here, A is the translation component of the joint's origin and B = A + l
    A = np.dot(R.T, joint_transform[0:3, -1])
    m = np.cross(A, l)

    return np.concatenate((l, m), axis=0)


def get_joint_ranges(joint):  # -> tuple[list, list]:
    """
    Computes the joint limits (ranges) for a given joint.

    Depending on the alignment (chirality) of the joint axis, the function extracts the rotational and
    translational limits from the joint's limit structure. For a joint with a positive chirality, the limits
    are taken directly; for a negative chirality, they are negated.

    Args:
        joint: An object representing a joint, which must contain the attributes 'axis' and 'limit'.
               The 'limit' attribute is expected to have 'lower_ang', 'upper_ang', 'lower_lin', and 'upper_lin'.

    Returns:
        tuple: A tuple (r_limits, p_limits) where:
            - r_limits (list): A list containing the lower and upper angular limits.
            - p_limits (list): A list containing the lower and upper linear (translational) limits.
    """
    chir = np.dot(joint.axis, np.abs(joint.axis))
    if chir >= 0:
        r_limits = [joint.limit.lower_ang, joint.limit.upper_ang]
        p_limits = [joint.limit.lower_lin, joint.limit.upper_lin]
    else:
        r_limits = [-joint.limit.upper_ang, -joint.limit.lower_ang]
        p_limits = [-joint.limit.upper_lin, -joint.limit.lower_lin]

    return r_limits, p_limits


def are_parallel(axis1, axis2):
    """
    Determines if two axes are parallel.

    This function checks if two normalized axes are parallel (pointing in the same or opposite direction)
    within a small tolerance. It returns 1 if the axes are parallel in the same direction, -1 if they are
    anti-parallel, and 0 otherwise.

    Args:
        axis1 (np.ndarray): The first normalized 3D vector.
        axis2 (np.ndarray): The second normalized 3D vector.

    Returns:
        int: 1 if axes are parallel (dot product close to 1), -1 if anti-parallel (dot product close to -1),
             or 0 if not parallel.
    """
    # Simple parallel check assuming axes are normalized
    if np.isclose(np.dot(axis1, axis2), 1.0, atol=1e-3):
        return 1
    elif np.isclose(np.dot(axis1, axis2), -1.0, atol=1e-3):
        return -1
    else:
        return 0


def generate_manifold_mesh(
    mesh_path,
    manifold_mesh_path,
    manifold_exec_path,
    target_number_of_triangles,
) -> None:
    """
    Generates a manifold version of a mesh using an external executable.

    This function calls an external mesh processing tool (via subprocess) to generate a manifold mesh
    from the input mesh. Supported formats include .ply, .stl, .obj, .off, etc. After processing,
    the mesh is read, smoothed, simplified, and cleaned before being saved back.

    Args:
        mesh_path (str): Path to the input mesh file.
        manifold_mesh_path (str): Path where the manifold mesh will be saved.
        manifold_exec_path (str): Path to the external executable used for generating the manifold mesh.
        target_number_of_triangles (int): The desired maximum number of triangles in the output mesh.

    Returns:
        None
    """
    # Load mesh. Supported formats: .ply, .stl, .obj, .off, etc.
    if mesh_path.endswith(("obj", "ply", "stl", "3ds", "dae", "off")):
        subprocess.run(
            [manifold_exec_path] + [mesh_path, manifold_mesh_path, str(50000)],
            stdout=subprocess.PIPE,
            text=True,
        )
    else:
        raise Exception(f"Model format not supported: {mesh_path}")

    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(manifold_mesh_path)

    # Perform mesh processing operations
    mesh = (
        mesh.filter_smooth_taubin(number_of_iterations=100)
        .filter_smooth_laplacian(number_of_iterations=10)
        .compute_triangle_normals()
        .compute_vertex_normals()
        .simplify_quadric_decimation(
            target_number_of_triangles=target_number_of_triangles
        )
        .filter_smooth_simple(number_of_iterations=1)
        .remove_degenerate_triangles()
        .remove_duplicated_triangles()
        .remove_duplicated_vertices()
        .remove_non_manifold_edges()
        .remove_unreferenced_vertices()
    )

    # Save the processed mesh
    o3d.io.write_triangle_mesh(manifold_mesh_path, mesh, write_vertex_normals=True)


def get_link_index(link_string) -> int:
    """
    Extracts the numeric index from a link string.

    The link string is expected to follow a convention such as 'link_<index>'.
    This function uses a regular expression to extract the index number.

    Args:
        link_string (str): The string representing the link name (e.g., 'link_0').

    Returns:
        int: The extracted index as an integer.
    """
    match = re.search(r"link_(\d+)", link_string)
    if match:
        return int(match.group(1))
    else:
        raise NameError(
            "Link name are expected to follow a convention of type 'link_<index>' such as 'link_0'."
        )


def process_folder(
    urdf_path,
    output_path,
    number_point_samples,
    manifold_exec_path=None,
    target_number_of_triangles=10000,
    regenerate_manifold=False,
    nap_graph_data_path=None,
):
    """
    Processes a folder containing a URDF file to generate graph dataset components.

    This function parses a URDF file to extract robot and joint information, aggregates visual meshes,
    computes transformation and bounding box data, exports meshes (in OFF and manifold formats),
    samples point clouds from the meshes, and assembles node and edge feature dictionaries for graph construction.
    Finally, it saves the processed data and generates a MuJoCo scene file for visualization/debugging.

    Args:
        urdf_path (str): Path to the URDF file to be processed.
        output_path (str): Directory where processed outputs (meshes, graph data, MuJoCo XML) will be saved.
        number_point_samples (int): Number of points to sample on the surface of each rigid body.
        manifold_exec_path (str, optional): Path to the external Manifold mesh generator executable.
                                              Defaults to None.
        target_number_of_triangles (int, optional): Target maximum number of triangles for the manifold mesh.
                                                    Defaults to 10000.
        regenerate_manifold (bool, optional): If True, forces regeneration of the manifold mesh even if it exists.
                                              Defaults to False.
        nap_graph_data_path (str, optional): Path to existing NAP graph data for consistency checking.
                                             Defaults to None.

    Returns:
        None
    """
    # Parse the dataset files
    joint_types = set(["prismatic", "revolute", "continuous"])
    robot, _, success_parse = load_urdf(urdf_path, False, False)

    if success_parse:
        nodes_feature_list = []
        edges_feature_list = []
        midgard_joints = []
        processed_links = set()
        processed_virtual_links = set()
        mesh_offset_dict = {}
        mesh_index_dict = {}

        # Loop through the URDF joints (i.e. no screw joints yet)
        # part_index = 0
        for urdf_joint in robot.joints:

            # Check that the joint type is valid
            if urdf_joint.joint_type in joint_types:

                # If one the connected links is a "virtual" helper link used to emulate a screw joint
                parent_link = urdf_joint.parent
                child_link = urdf_joint.child

                # Virtual links are those having no visuals
                has_virtual_parent_link = not robot._link_map[parent_link].visuals
                has_virtual_child_link = not robot._link_map[child_link].visuals

                # If the selected joint is connected to a virtual link
                if has_virtual_parent_link or has_virtual_child_link:

                    if (
                        parent_link not in processed_virtual_links
                        and child_link not in processed_virtual_links
                    ):

                        if has_virtual_child_link:

                            other_joint = [
                                j for j in robot.joints if j.parent == urdf_joint.child
                            ]
                            other_joint = other_joint[0]
                            parent_link = urdf_joint.parent
                            child_link = other_joint.child
                            parent_origin = urdf_joint.origin
                            parent_axis = urdf_joint.axis
                            processed_virtual_links.add(urdf_joint.child)
                            if urdf_joint.child not in processed_links:
                                processed_links.add(urdf_joint.child)

                        elif has_virtual_parent_link:

                            other_joint = [
                                j for j in robot.joints if j.child == urdf_joint.parent
                            ]
                            other_joint = other_joint[0]
                            parent_link = other_joint.parent
                            child_link = urdf_joint.child
                            parent_origin = other_joint.origin
                            parent_axis = other_joint.axis
                            processed_virtual_links.add(urdf_joint.parent)
                            if urdf_joint.parent not in processed_links:
                                processed_links.add(urdf_joint.parent)

                        if other_joint and are_parallel(
                            urdf_joint.axis, other_joint.axis
                        ):

                            # Joints share a helper link and are parallel: candidate for fusion into a screw joint
                            urdf_anti_axis = (
                                np.dot(urdf_joint.axis, np.abs(urdf_joint.axis)) < 0
                            )
                            other_anti_axis = (
                                np.dot(other_joint.axis, np.abs(other_joint.axis)) < 0
                            )
                            direct = not (urdf_anti_axis and other_anti_axis)
                            if urdf_joint.joint_type == "prismatic":
                                parent_lower_lin = (
                                    urdf_joint.limit.lower
                                    if not urdf_anti_axis
                                    else -urdf_joint.limit.upper
                                )
                                parent_upper_lin = (
                                    urdf_joint.limit.upper
                                    if not urdf_anti_axis
                                    else -urdf_joint.limit.lower
                                )
                                parent_lower_ang = 0.0
                                parent_upper_ang = 0.0
                            elif urdf_joint.joint_type == "revolute":
                                parent_lower_lin = 0.0
                                parent_upper_lin = 0.0
                                parent_lower_ang = (
                                    urdf_joint.limit.lower
                                    if not urdf_anti_axis
                                    else -urdf_joint.limit.upper
                                )
                                parent_upper_ang = (
                                    urdf_joint.limit.upper
                                    if not urdf_anti_axis
                                    else -urdf_joint.limit.lower
                                )
                            else:  # continuous
                                parent_lower_lin = 0.0
                                parent_upper_lin = 0.0
                                parent_lower_ang = (
                                    0.0 if not urdf_anti_axis else -2 * np.pi
                                )
                                parent_upper_ang = (
                                    2 * np.pi if not urdf_anti_axis else 0.0
                                )

                            if other_joint.joint_type == "prismatic":
                                child_lower_lin = (
                                    other_joint.limit.lower
                                    if not other_anti_axis
                                    else -other_joint.limit.upper
                                )
                                child_upper_lin = (
                                    other_joint.limit.upper
                                    if not other_anti_axis
                                    else -other_joint.limit.lower
                                )
                                child_lower_ang = 0.0
                                child_upper_ang = 0.0
                            elif other_joint.joint_type == "revolute":
                                child_lower_lin = 0.0
                                child_upper_lin = 0.0
                                child_lower_ang = (
                                    other_joint.limit.lower
                                    if not other_anti_axis
                                    else -other_joint.limit.upper
                                )
                                child_upper_ang = (
                                    other_joint.limit.upper
                                    if not other_anti_axis
                                    else -other_joint.limit.lower
                                )
                            else:  # continuous
                                child_lower_lin = 0.0
                                child_upper_lin = 0.0
                                child_lower_ang = (
                                    0.0 if not other_anti_axis else -2 * np.pi
                                )
                                child_upper_ang = (
                                    2 * np.pi if not other_anti_axis else 0.0
                                )

                            midgard_joints.append(
                                midgard_Joint(
                                    other_joint.name,
                                    "screw",
                                    parent_link,
                                    child_link,
                                    origin=parent_origin,
                                    axis=parent_axis,
                                    limit=midgard_JointLimit(
                                        lower_lin=parent_lower_lin + child_lower_lin,
                                        upper_lin=parent_upper_lin + child_upper_lin,
                                        lower_ang=parent_lower_ang + child_lower_ang,
                                        upper_ang=parent_upper_ang + child_upper_ang,
                                    ),
                                    direct=direct,
                                )
                            )

                else:
                    urdf_anti_axis = (
                        np.dot(urdf_joint.axis, np.abs(urdf_joint.axis)) < 0
                    )
                    direct = not urdf_anti_axis
                    # Consider valid joint parameters for further computations
                    if urdf_joint.joint_type == "prismatic":
                        lower_lin = (
                            urdf_joint.limit.lower
                            if not urdf_anti_axis
                            else -urdf_joint.limit.upper
                        )
                        upper_lin = (
                            urdf_joint.limit.upper
                            if not urdf_anti_axis
                            else -urdf_joint.limit.lower
                        )
                        lower_ang = 0.0
                        upper_ang = 0.0
                    elif urdf_joint.joint_type == "revolute":
                        lower_lin = 0.0
                        upper_lin = 0.0
                        lower_ang = (
                            urdf_joint.limit.lower
                            if not urdf_anti_axis
                            else -urdf_joint.limit.upper
                        )
                        upper_ang = (
                            urdf_joint.limit.upper
                            if not urdf_anti_axis
                            else -urdf_joint.limit.lower
                        )
                    else:  # continuous
                        lower_lin = 0.0
                        upper_lin = 0.0
                        lower_ang = 0.0 if not urdf_anti_axis else -2 * np.pi
                        upper_ang = 2 * np.pi if not urdf_anti_axis else 0.0

                    midgard_joints.append(
                        midgard_Joint(
                            urdf_joint.name,
                            urdf_joint.joint_type,
                            parent_link,
                            child_link,
                            origin=urdf_joint.origin,
                            axis=urdf_joint.axis,
                            limit=midgard_JointLimit(
                                lower_lin=lower_lin,
                                upper_lin=upper_lin,
                                lower_ang=lower_ang,
                                upper_ang=upper_ang,
                            ),
                            direct=direct,
                        )
                    )

                for node_name in [
                    child_link,
                    parent_link,
                ]:  # TODO: investigate influence of [child_link, parent_link] vs [parent_link, child_link]

                    if node_name not in processed_links:

                        processed_links.add(node_name)
                        part_index = get_link_index(node_name)

                        # Get the link object corresponding to the corresponding node
                        link_object = robot._link_map[node_name]

                        # Compute mesh-related features
                        (
                            abs_center,
                            bbox_L,
                            mesh_T_lm_list,
                            agg_trimesh,
                        ) = recenter_aggregate_mesh(link_object)

                        # Save aggregated link meshes
                        combined_mesh = (
                            np.asarray(agg_trimesh.vertices),
                            np.asarray(agg_trimesh.faces),
                        )
                        mesh_dir = os.path.join(output_path, "meshes")
                        mesh_name = str(part_index) + ".off"
                        mesh_path = os.path.join(mesh_dir, mesh_name)
                        os.makedirs(mesh_dir, exist_ok=True)
                        export_off(combined_mesh[0], combined_mesh[1], mesh_path)

                        # Optionally generate a manifold mesh here...
                        manifold_mesh_dir = os.path.join(output_path, "manifold_meshes")
                        manifold_mesh_name = str(part_index) + ".obj"
                        manifold_mesh_path = os.path.join(
                            manifold_mesh_dir, manifold_mesh_name
                        )
                        os.makedirs(manifold_mesh_dir, exist_ok=True)
                        if manifold_exec_path and (
                            not os.path.exists(manifold_mesh_path)
                            or regenerate_manifold
                        ):
                            generate_manifold_mesh(
                                mesh_path,
                                manifold_mesh_path,
                                manifold_exec_path,
                                target_number_of_triangles,
                            )

                        # Sample number_point_samples points on the aggregated mesh
                        pcl = sample_mesh(agg_trimesh, number_point_samples)

                        # Compute the Oriented Bounding Box (OBB) and related quantities
                        mesh = o3d.io.read_triangle_mesh(manifold_mesh_path)
                        if mesh.is_empty():
                            raise ValueError("Loaded mesh is empty or corrupted.")
                        min_obb = compute_minimum_oriented_bounding_box(mesh)
                        obb_type = "box"
                        obb_center = min_obb.center
                        obb_quat = quaternion.as_float_array(
                            quaternion.from_rotation_matrix(min_obb.R)
                        )
                        obb_size = min_obb.extent / 2

                        # List of the names of the different meshes
                        mesh_name_list = [
                            visual.geometry.mesh.filename.split("/")[-1].split(".")[0]
                            for visual in link_object.visuals
                        ]

                        # Store the offset transform for later use in the edge features computation
                        mesh_offset_dict[node_name] = mesh_T_lm_list[0]
                        mesh_index_dict[node_name] = part_index

                        # Assemble the node feature dictionary
                        node_feature_dic = {}
                        node_feature_dic["names"] = [node_name]
                        node_feature_dic["mesh_name_list"] = mesh_name_list
                        node_feature_dic["mesh_T_lm_list"] = mesh_T_lm_list
                        node_feature_dic["agg_mesh"] = combined_mesh
                        node_feature_dic["abs_center"] = abs_center
                        node_feature_dic["bbox_L"] = bbox_L
                        node_feature_dic["pcl"] = pcl
                        node_feature_dic["obb_type"] = obb_type
                        node_feature_dic["obb_center"] = (
                            f"{obb_center[0]:.5f} {obb_center[1]:.5f} {obb_center[2]:.5f}"
                        )
                        node_feature_dic["obb_quat"] = (
                            f"{obb_quat[0]:.5f} {obb_quat[1]:.5f} {obb_quat[2]:.5f} {obb_quat[3]:.5f}"
                        )
                        node_feature_dic["obb_size"] = (
                            f"{obb_size[0]:.5f} {obb_size[1]:.5f} {obb_size[2]:.5f}"
                        )
                        node_feature_dic["idx"] = part_index
                        nodes_feature_list.append(node_feature_dic)

                        # Increment part index
                        # part_index = part_index + 1

        for joint in midgard_joints:
            # Transforms
            Tij_src_j1 = (
                np.linalg.inv(robot._link_map[joint.parent].visuals[0].origin)
                @ mesh_offset_dict[joint.parent]
                @ joint.origin
            )
            # Centered parent frame to joint
            Tij_j2_dst = robot._link_map[joint.child].visuals[0].origin @ np.linalg.inv(
                mesh_offset_dict[joint.child]
            )  # Joint to centered child frame
            T0 = mesh_offset_dict[joint.parent] @ np.linalg.inv(
                mesh_offset_dict[joint.child]
            )
            Tji_src_j1 = np.linalg.inv(Tij_j2_dst)
            Tji_j2_dst = np.linalg.inv(Tij_src_j1)

            # Determine the Plücker joint parametrizations for the parent->child and for the child->parent configurations
            plucker_ij = compute_plucker_coordinates(joint.axis, Tij_src_j1)
            plucker_ji = compute_plucker_coordinates(-joint.axis, Tji_src_j1)

            # Parametrize the Parent->Child edge configurations
            e0 = {}
            e0["src_name"] = joint.parent
            e0["dst_name"] = joint.child
            e0["axis"] = joint.axis
            e0["T_src_j1"] = Tij_src_j1
            e0["T_j2_dst"] = Tij_j2_dst
            e0["src_ind"] = mesh_index_dict[joint.parent]
            e0["dst_ind"] = mesh_index_dict[joint.child]
            e0["plucker"] = plucker_ij
            e0["T0"] = T0

            # Parametrize the Child->Parent edge configuration
            e1 = {}
            e1["src_name"] = joint.child
            e1["dst_name"] = joint.parent
            e1["src_ind"] = mesh_index_dict[joint.child]
            e1["dst_ind"] = mesh_index_dict[joint.parent]
            e1["axis"] = -joint.axis
            e1["T_src_j1"] = Tji_src_j1
            e1["T_j2_dst"] = Tji_j2_dst
            e1["plucker"] = plucker_ji
            e1["T0"] = np.linalg.inv(T0)

            # Get the joint range
            r_limits, p_limits = get_joint_ranges(joint)

            # Assemble the edge feature dictionary
            edge_feature_dic = {}
            edge_feature_dic["name"] = joint.name
            edge_feature_dic["r_limits"] = r_limits
            edge_feature_dic["p_limits"] = p_limits
            edge_feature_dic["e0"] = e0
            edge_feature_dic["e1"] = e1
            edges_feature_list.append(edge_feature_dic)

        # Sort the node and edge featured based on their name
        # Function to extract the numerical part from the string
        sorted_nodes_feature_list = sorted(
            nodes_feature_list, key=lambda x: int(x["names"][0].split("_")[1])
        )
        sorted_edges_feature_list = sorted(
            edges_feature_list, key=lambda x: int(x["name"].split("_")[1])
        )

        # Create a new list with None placeholders based on the maximum index found
        max_index = max(
            int(link["names"][0].split("_")[1]) for link in sorted_nodes_feature_list
        )
        ordered_links = [None] * (max_index + 1)
        # Place each item in the correct index position
        for link in sorted_nodes_feature_list:
            index = int(link["names"][0].split("_")[1])
            ordered_links[index] = link

        # Check for discrepancies
        discrepancies = [
            index for index, link in enumerate(ordered_links) if link is None
        ]

        # Report discrepancies
        if discrepancies:
            print(
                f"Discrepancies detected! Missing links for the following indices: {discrepancies}"
            )

        # Save processed data in compressed format
        graph_path = os.path.join(output_path, "graph.npz")
        np.savez(
            graph_path,
            V=np.array(sorted_nodes_feature_list),
            E=np.array(sorted_edges_feature_list),
        )

        # Split the path by '/'
        if os.path.isdir(nap_graph_data_path):
            path_parts = urdf_path.split("/")
            asset_name = path_parts[-2]
            nap_graph_path = os.path.join(nap_graph_data_path, asset_name + ".npz")
            path_parts = urdf_path.split(os.sep)
            raw_data_index = path_parts.index("mobility.urdf")
            code = path_parts[raw_data_index - 1]
            check_graph_consistency_with_nap(graph_path, nap_graph_path, code)

        # Genrate a MuJoCo file for visualization/debug purposes
        generate_mujoco_scene_from_VE(
            os.path.join(output_path, "mujoco_gt.xml"),
            V=nodes_feature_list,
            E=edges_feature_list,
        )


def main(argv) -> None:
    """
    Main entry point for processing the raw PartNet Mobility dataset to generate a graph dataset.

    The function parses command-line arguments to determine dataset directories, manifold mesh parameters,
    and point sampling settings. It then loads metadata, constructs file paths for each asset, and uses
    parallel processing to invoke the process_folder function for each asset. Finally, it prints a completion
    message.

    Args:
        argv (list of str): List of command-line arguments.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Parse the raw Partnet Mobility data to generate a proper graph dataset."
    )
    parser.add_argument(
        "dataset_directory",
        type=str,
        help="Path to the directory containing the meshes.",
    )
    parser.add_argument(
        "--manifold_exec_path",
        default="",
        type=str,
        help="Path to the Manifold mesh generator.",
    )
    parser.add_argument(
        "--point_samples",
        default=10000,
        type=int,
        help="Number of points to be sampled on the surface of each rigid body.",
    )
    parser.add_argument(
        "--target_number_of_triangles",
        default=10000,
        type=int,
        help="Max number of triangles in the output mesh.",
    )
    args = parser.parse_args(argv)

    # Sanity checks
    data_directory = os.path.join(args.dataset_directory, "data")
    metadata_directory = os.path.join(args.dataset_directory, "metadata")
    raw_data_directory = os.path.join(args.dataset_directory, "raw")
    assert os.path.exists(
        metadata_directory
    ), "The folder supposed to contain the metadata does not exist. Aborting..."
    assert os.path.exists(
        raw_data_directory
    ), "The folder supposed to contain the raw data does not exist. Aborting..."

    number_point_samples = args.point_samples

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
    path_dict = {key: [] for key in assets_ids}

    # Loop through each asset id
    for cat in assets_ids:
        for code in assets_ids[cat]:
            urdf_path = os.path.join(raw_data_directory, code, "mobility.urdf")
            output_path = os.path.join(data_directory, cat, code)
            path_dict[cat].append((urdf_path, output_path))

    print("Generating graph dataset...")
    for cat_id in info["all_cats"]:

        # Mesh files for a specific category of asset
        print(f"Processing assets of type: {cat_id}...")
        paths = path_dict[cat_id]

        with tqdm_joblib(tqdm(desc="Processing raw dataset", total=len(paths))):
            Parallel(n_jobs=os.cpu_count())(
                delayed(process_folder)(
                    urdf_path,
                    output_path,
                    number_point_samples,
                    args.manifold_exec_path,
                    args.target_number_of_triangles,
                    regenerate_manifold=False,
                    nap_graph_data_path="",
                    # nap_graph_data_path="/Users/quentin/Documents/Datasets/partnet_mobility/partnet_mobility_graph_v4/",
                )
                for (urdf_path, output_path) in paths
            )

    print("Graph dataset generated!")


# Main execution logic
if __name__ == "__main__":
    main(sys.argv[1:])
