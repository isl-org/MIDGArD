import copy
import networkx as nx
import numpy as np
import imageio
import os
import open3d as o3d
import quaternion
import sys
import trimesh
from PIL import Image
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from defusedxml.minidom import parseString
import xml.etree.ElementTree as ElementTree
from core.models.shape_generator.condition.gat import get_graph_bb
from core.utils.data_utils import write_json, create_bbox_3d
from core.utils.image_processor import ImageProcessor, ImgProcCfg

FLIP_AXES_CORRECTION = True

def resolve_range_(joint_label, rlim, plim):
    """
    Resolves the rotational and positional joint limits based on the joint type indicated
    by the joint label.

    Args:
        joint_label (np.ndarray): A vector where the index of the maximum value indicates the joint type.
                                  - Index 0: Screw joint.
                                  - Index 1: Revolute joint.
                                  - Index 2: Prismatic joint.
        rlim (np.ndarray): A numpy array representing the rotational limits of the joint.
        plim (np.ndarray): A numpy array representing the positional (translational) limits of the joint.

    Returns:
        tuple: A tuple (rlim, plim) where:
            - rlim (np.ndarray): The updated rotational limits (set to zero for prismatic joints).
            - plim (np.ndarray): The updated positional limits (set to zero for revolute joints).
    """
    label = np.argmax(joint_label)
    if label == 0:  # Screw
        pass
    elif label == 1:  # Revolute
        plim *= 0
    else:  # Prismatic
        rlim *= 0

    return rlim, plim


def plucker_to_mujoco_joint_(l, m, plim, rlim, threshold=1e-2):
    """
    Converts Plücker coordinates into one or two MuJoCo joint specifications.

    Args:
        l (np.ndarray): A unit direction vector derived from the Plücker coordinates.
        m (np.ndarray): The moment vector derived from the Plücker coordinates.
        plim (np.ndarray): An array defining the translational limits of the joint.
        rlim (np.ndarray): An array defining the rotational limits of the joint.
        threshold (float, optional): A small value used to decide if a limit range is effectively zero.
                                     Defaults to 1e-2.

    Returns:
        list: A list containing one or two dictionaries, each representing a MuJoCo joint specification.
              - For a helicoidal joint (both limits non-negligible), two joint specifications are returned:
                one for the sliding component and one for the hinge component.
              - For purely prismatic or revolute joints, a single joint specification is returned.
    """

    # Ensure that l is a unit vector
    assert abs(np.linalg.norm(l) - 1.0) < 1e-4

    offset = np.cross(l, m)

    joint = {
        "axis": l,
        "damping": "2",
        "armature": "0.01",
        "frictionloss": "0.2",
        "offset": offset,
    }

    if (
        abs(rlim[0] - rlim[1]) >= threshold and abs(plim[0] - plim[1]) >= threshold
    ):  # Helicoidal joint
        joint_1 = copy.deepcopy(joint)
        joint_2 = copy.deepcopy(joint)
        joint_1["type"] = "slide"
        joint_1["range"] = plim
        joint_2["type"] = "hinge"
        joint_2["range"] = rlim
        return [joint_1, joint_2]
    elif abs(rlim[0] - rlim[1]) < threshold:  # Prismatic joint
        joint["type"] = "slide"
        joint["range"] = plim
        return [joint]
    elif abs(plim[0] - plim[1]) < threshold:  # Revolute joint
        joint["type"] = "hinge"
        joint["range"] = rlim
        return [joint]
    else:  # Unknown
        raise NotImplementedError()


def get_unoriented_bb(vertices: np.ndarray):
    """
    Computes the center and half-extents of an axis-aligned bounding box for a given set of vertices.

    Args:
        vertices (np.ndarray): An array of shape (n, 3) containing the 3D coordinates of the mesh vertices.

    Returns:
        tuple: A tuple containing:
            - center (np.ndarray): A 1D array with the 3D coordinates of the bounding box center.
            - size (np.ndarray): A 1D array with the half-size (extent) of the bounding box along each axis.
    """
    min_verts = np.min(vertices, axis=0)
    size = (np.max(vertices, axis=0) - min_verts) / 2
    center = min_verts + size
    return center, size


def scale_mesh(vertices: np.ndarray, desired_pose: dict):
    """
    Scales and translates mesh vertices based on a specified desired pose.

    Args:
        vertices (np.ndarray): An array of shape (n, 3) representing the original vertices of the mesh.
        desired_pose (dict): A dictionary specifying the target pose with the following keys:
            - "size": The desired size (scaling factor) of the mesh.
            - "pos": The desired position (translation vector) of the mesh center.

    Returns:
        np.ndarray: An array of the transformed vertices after scaling and translation.
    """
    current_center, current_size = get_unoriented_bb(vertices)
    centered_vertices = vertices - current_center  # center
    vertices_canonical = centered_vertices / current_size

    vertices_desired = (
        vertices_canonical * desired_pose["size"] / 2 + desired_pose["pos"]
    )

    return vertices_desired


def add_body_VE_(V, E, idx, parent_xml, parent_node_idx=None) -> None:
    """
    Recursively adds bodies and joints to a MuJoCo XML tree based on provided vertex and edge data.

    This function traverses a graph structure representing the kinematic tree of a model. It adds
    bodies (with associated visual and collision geometry) and joints by parsing node (V) and edge (E)
    data. The recursion ensures that all child bodies connected to the current node are processed
    and attached appropriately in the XML hierarchy.

    Args:
        V (list): A list of dictionaries, each representing node (body) data. Each dictionary is expected
                  to contain at least the keys "idx", "obb_type", "obb_center", "obb_quat", "obb_size",
                  and bounding box information (e.g., 'bbox_L').
        E (list): A list of dictionaries, each representing edge data between nodes. Each edge dictionary
                  should contain connection details under the key "e0", which includes:
                      - "src_ind": Source node index.
                      - "dst_ind": Destination node index.
                      - "plucker": The Plücker coordinates.
                      - "T0": A transformation matrix.
                      - "p_limits": Positional limits.
                      - "r_limits": Rotational limits.
        idx (int): The index of the current node to process.
        parent_xml (xml.etree.ElementTree.Element): The XML element under which the current body's XML
                                                    representation will be added.
        parent_node_idx (Optional[int], optional): The index of the parent node. Defaults to None,
                                                     indicating that the current node is the root.

    Returns:
        None: The function updates the provided XML tree in place.
    """

    # Dummy temp placeholders
    node_data = [v for v in V if v["idx"] == idx][0]
    color = np.array([0.5, 0.5, 0.5, 1.0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])

    if parent_node_idx == None:
        xyz = np.array([0.0, 0.0, 0.5])
        body = ElementTree.SubElement(
            parent_xml,
            "body",
            name=f"body_{idx}",
            pos=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
            quat=f"{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}",
        )
        ElementTree.SubElement(body, "freejoint")
    else:
        edge_data = [
            e
            for e in E
            if (
                (e["e0"]["src_ind"] == parent_node_idx and e["e0"]["dst_ind"] == idx)
                or (e["e0"]["src_ind"] == idx and e["e0"]["dst_ind"] == parent_node_idx)
            )
        ][0]

        plucker = edge_data["e0"]["plucker"]
        if edge_data["e0"]["src_ind"] == idx:
            # The edge direction matches the src->dst order
            c = 0
            l, m = -plucker[:3], -plucker[3:]
            _T0 = np.linalg.inv(edge_data["e0"]["T0"])
        else:
            # The edge direction is dst->src, so we need to invert the transform
            l, m = plucker[:3], plucker[3:]
            # edge_data["T_src_dst"]
            _T0 = edge_data["e0"]["T0"]
            c = 1

        plim, rlim = edge_data["p_limits"], edge_data["r_limits"]
        mujoco_joints = plucker_to_mujoco_joint_(l, m, plim, rlim)
        clear_config = False

        for i, mujoco_joint in enumerate(mujoco_joints):
            axis = mujoco_joint["axis"]
            joint_range = mujoco_joint["range"]

            if clear_config:
                xyz *= 0
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                xyz = _T0[0:3, -1]

            if (
                len(mujoco_joints) - i > 1
            ):  # Add dumy bodies to account for complex joints
                clear_config = True
                body = ElementTree.SubElement(
                    parent_xml,
                    "body",
                    name=f"dummy_body_{idx}_{i}",
                    pos=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
                    quat=f"{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}",
                )
                xyz = -c * _T0[0:3, 3] + mujoco_joint["offset"]
                ElementTree.SubElement(
                    body,
                    "joint",
                    name=f"dummy_joint_{i}_{idx}",
                    type=mujoco_joint["type"],
                    axis=f"{axis[0]} {axis[1]} {axis[2]}",
                    pos=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
                    range=f"{joint_range[0]} {joint_range[1]}",
                    damping=mujoco_joint["damping"],
                    armature=mujoco_joint["armature"],
                    frictionloss=mujoco_joint["frictionloss"],
                )

                geom_attributes = {
                    "name": f"dummy_geom_{i}_{idx}",
                    "type": "sphere",
                    "size": f"{0.01}",
                    "class": "visual",
                    "rgba": f"{0.0} {0.0} {0.0} {0.0}",
                }
                ElementTree.SubElement(body, "geom", geom_attributes)

                parent_xml = body
            else:
                body = ElementTree.SubElement(
                    parent_xml,
                    "body",
                    name=f"body_{idx}",
                    pos=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
                    quat=f"{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}",
                )
                xyz = -c * _T0[0:3, 3] + mujoco_joint["offset"]
                ElementTree.SubElement(
                    body,
                    "joint",
                    name=f"joint_{parent_node_idx}_{idx}",
                    type=mujoco_joint["type"],
                    axis=f"{axis[0]} {axis[1]} {axis[2]}",
                    pos=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
                    range=f"{joint_range[0]} {joint_range[1]}",
                    damping=mujoco_joint["damping"],
                    armature=mujoco_joint["armature"],
                    frictionloss=mujoco_joint["frictionloss"],
                )

    # Show mesh
    geom_attributes = {
        "name": f"geom_{idx}_vis",
        "mesh": f"mesh_{idx}",
        "class": "visual",
        "rgba": f"{color[0]} {color[1]} {color[2]} {1}",
    }
    ElementTree.SubElement(body, "geom", geom_attributes)
    geom_attributes = {
        "name": f"geom_{idx}_col",
        "mesh": f"mesh_{idx}",
        "class": "collision",
    }
    ElementTree.SubElement(body, "geom", geom_attributes)

    # Show OBB
    geom_attributes = {
        "name": f"geom_{idx}_obb",
        "type": node_data["obb_type"],
        "pos": node_data["obb_center"],
        "quat": node_data["obb_quat"],
        "size": node_data["obb_size"],
        "class": "visual",
        "rgba": f"{0.0} {1.0} {0.0} {0.25}",
    }
    ElementTree.SubElement(body, "geom", geom_attributes)

    # Show AABB
    geom_attributes = {
        "name": f"geom_{idx}_aabb",
        "type": "box",
        # "pos": f"{node_data['abs_center'][0]} {node_data['abs_center'][1]} {node_data['abs_center'][2]}",
        "size": f"{node_data['bbox_L'][0]} {node_data['bbox_L'][1]} {node_data['bbox_L'][2]}",
        "class": "visual",
        "rgba": f"{1.0} {0.0} {0.0} {0.25}",
    }
    ElementTree.SubElement(body, "geom", geom_attributes)

    # Process the children of this node
    neighbors = []
    for e in E:
        if e["e0"]["src_ind"] == idx and not (idx in neighbors):
            neighbors.append(e["e0"]["dst_ind"])
        if e["e0"]["dst_ind"] == idx and not (idx in neighbors):
            neighbors.append(e["e0"]["src_ind"])

    for neighbor_idx in neighbors:  # graph.neighbors(idx):
        if neighbor_idx != idx:  # Avoid adding self
            if parent_node_idx == None or neighbor_idx != parent_node_idx:
                # Recursively add the child body
                add_body_VE_(V, E, neighbor_idx, body, idx)


def generate_mujoco_scene_from_VE(scene_path, V, E, white_background=False) -> None:
    """
    Generates a MuJoCo XML scene file from vertex and edge data.

    This function creates a structured XML file for a MuJoCo simulation environment.
    It defines compiler options, simulation settings, default visual and collision properties,
    assets (meshes and textures), and the world body (lighting, ground plane, and kinematic tree).
    The scene is built using vertex (V) and edge (E) data, and the kinematic tree is processed
    recursively via the add_body_VE_ function.

    Args:
        scene_path (str): The file path where the XML scene file will be saved.
        V (list): A list of dictionaries representing node (body) data. Each node must include
                  keys such as "idx", "bbox_L", and other properties used for visualization.
        E (list): A list of dictionaries representing edge (joint) data between nodes. Each edge
                  should contain the key "e0" with joint parameters, transformation matrices, and limits.
        white_background (bool, optional): If True, configures the scene with a white background,
                                           affecting textures and visual settings. Defaults to False.

    Returns:
        None
    """
    # Create the root element of the XML file with the model attribute
    mujoco = ElementTree.Element("mujoco", {"model": "midgard scene"})

    # Specify memory size for the simulation
    ElementTree.SubElement(mujoco, "size", {"memory": "50M"})

    # Define statistics for simulation analysis
    ElementTree.SubElement(mujoco, "statistic", {"center": "0 0 .3", "extent": "1.2"})

    # Add compiler settings to the XML
    ElementTree.SubElement(
        mujoco,
        "compiler",
        {
            "angle": "radian",
            "balanceinertia": "true",
            "autolimits": "true",
            "fusestatic": "true",
            # "convexhull": "false",
        },
    )

    # Add simulation options such as timestep, integrator, solver, etc.
    ElementTree.SubElement(
        mujoco,
        "option",
        {
            "timestep": "0.001",
            "integrator": "implicitfast",
            "solver": "Newton",
            "cone": "pyramidal",
        },
    )

    # Set up default properties
    default = ElementTree.SubElement(mujoco, "default")
    ElementTree.SubElement(default, "material", {"specular": "0", "shininess": "0.25"})

    # Define default visual properties for geometries
    visual_default = ElementTree.SubElement(default, "default", {"class": "visual"})
    ElementTree.SubElement(
        visual_default,
        "geom",
        {"group": "2", "type": "mesh", "contype": "0", "conaffinity": "0"},
    )

    # Define default collision properties for geometries
    collision_default = ElementTree.SubElement(
        default, "default", {"class": "collision"}
    )
    ElementTree.SubElement(
        collision_default,
        "geom",
        {
            "group": "3",
            "type": "mesh",
            "priority": "1",
            "solimp": "0.015 0.09 0.01",
            "condim": "6",
        },
    )

    visual = ElementTree.SubElement(mujoco, "visual")
    asset = ElementTree.SubElement(mujoco, "asset")
    if white_background:
        # Configure visual elements like lighting and color
        ElementTree.SubElement(visual, "rgba", {"haze": "1 1 1 0.5"})
        ElementTree.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

        # Add skybox texture
        ElementTree.SubElement(
            asset,
            "texture",
            {
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "1 1 1",
                "rgb2": "0.5 0.5 0.5",
                "width": "512",
                "height": "3072",
            },
        )
        # Add ground plane texture
        ElementTree.SubElement(
            asset,
            "texture",
            {
                "type": "2d",
                "name": "groundplane",
                "builtin": "checker",
                "mark": "edge",
                "rgb1": "1 1 1",
                "rgb2": "1 1 1",
                "markrgb": "1 1 1",
                "width": "300",
                "height": "300",
            },
        )
    else:
        # Configure visual elements like lighting and color
        ElementTree.SubElement(
            visual,
            "headlight",
            {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"},
        )
        ElementTree.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
        ElementTree.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

        # Add skybox texture
        ElementTree.SubElement(
            asset,
            "texture",
            {
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.3 0.5 0.7",
                "rgb2": "0 0 0",
                "width": "512",
                "height": "3072",
            },
        )
        # Add ground plane texture
        ElementTree.SubElement(
            asset,
            "texture",
            {
                "type": "2d",
                "name": "groundplane",
                "builtin": "checker",
                "mark": "edge",
                "rgb1": "0.2 0.3 0.4",
                "rgb2": "0.1 0.2 0.3",
                "markrgb": "0.8 0.8 0.8",
                "width": "300",
                "height": "300",
            },
        )

    # Define material for the ground plane
    ElementTree.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )

    # Asset meshes
    for node in V:
        # Extract mesh data
        part_index = str(node["idx"])

        # Store mesh path in the xml file
        ElementTree.SubElement(
            asset,
            "mesh",
            {
                # "class": "midgard",
                # "file": f"meshes/{part_index}.obj",
                "file": f"manifold_meshes/{part_index}.obj",
                "name": f"mesh_{part_index}",
            },
        )

    # Set up the world body, including lighting and the ground plane
    worldbody = ElementTree.SubElement(mujoco, "worldbody")
    # Light
    ElementTree.SubElement(
        worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"}
    )

    # Floor
    ElementTree.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "size": "0 0 0.05",
            "type": "plane",
            "material": "groundplane",
        },
    )

    # Process and add body elements to the XML
    # Assuming the root of your kinematic tree is named 'root'
    nodes_bbox = np.stack([d["bbox_L"] for d in V], 0)
    nodes_volume = nodes_bbox.prod(axis=-1) * 8
    root_node = V[nodes_volume.argmax()]["idx"]
    add_body_VE_(V, E, root_node, worldbody)

    # Write the XML tree to a file
    pretty_xml = prettify_xml_(mujoco)
    with open(scene_path, "w") as f:
        f.write(pretty_xml)


def add_body_G_(graph, node, parent_xml, parent_node=None) -> None:
    """
    Recursively adds bodies and joints to a MuJoCo XML tree based on graph data.

    This function traverses a graph (networkx or similar) representing the kinematic tree of a model.
    It adds bodies with visual and collision geometry and configures joints using Plücker coordinates.
    The function processes the current node and then recursively processes its neighbors.

    Args:
        graph (networkx.Graph or similar): A graph structure where nodes represent bodies and edges contain
                                             joint information (e.g., Plücker coordinates, transformation matrices,
                                             joint limits, and labels).
        node (hashable): The current node identifier in the graph.
        parent_xml (xml.etree.ElementTree.Element): The XML element under which the current body's XML representation
                                                    will be added.
        parent_node (optional, hashable): The identifier of the parent node. Defaults to None, which indicates that
                                          the current node is the root.

    Returns:
        None
    """
    # Dummy temp placeholders
    node_data = graph.nodes[node]
    color = node_data["color"]
    bbox = np.abs(np.array(node_data["bbox"]))
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    quat_id = np.array([1.0, 0.0, 0.0, 0.0])

    if parent_node == None:
        xyz = np.array([0.0, 0.0, 0.5])
        body = ElementTree.SubElement(
            parent_xml,
            "body",
            name=f"body_{node}",
            pos=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
            quat=f"{quat_id [0]:.5f} {quat_id [1]:.5f} {quat_id [2]:.5f} {quat_id [3]:.5f}",
        )
        ElementTree.SubElement(body, "freejoint")
    else:
        assert graph.has_edge(parent_node, node)
        edge_data = graph.get_edge_data(parent_node, node)
        print(edge_data)
        plucker = edge_data["plucker"]
        if edge_data["src"] == node:
            # The edge direction matches the src->dst order
            c = 0
            l, m = -plucker[:3], -plucker[3:]
            _T0 = np.linalg.inv(edge_data["T_src_dst"])

        else:
            # The edge direction is dst->src, so we need to invert the transform
            l, m = plucker[:3], plucker[3:]
            _T0 = edge_data["T_src_dst"]
            c = 1

        plim, rlim = edge_data["plim"], edge_data["rlim"]
        if "joint_label" in edge_data:
            rlim, plim = resolve_range_(
                joint_label=edge_data["joint_label"], rlim=rlim, plim=plim
            )
        # l = _T0[0:3, 0:3].T @ l
        # m = _T0[0:3, 0:3].T @ m
        mujoco_joints = plucker_to_mujoco_joint_(l, m, plim, rlim)
        clear_config = False

        quat = quaternion.from_rotation_matrix(_T0[0:3, 0:3])
        quat = quaternion.as_float_array(quat)

        for i, mujoco_joint in enumerate(mujoco_joints):
            axis = mujoco_joint["axis"]
            joint_range = mujoco_joint["range"]

            if clear_config:
                xyz *= 0
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                xyz = _T0[0:3, -1]

            if (
                len(mujoco_joints) - i > 1
            ):  # Add dumy bodies to account for complex joints
                clear_config = True
                body = ElementTree.SubElement(
                    parent_xml,
                    "body",
                    name=f"dummy_body_{node}_{i}",
                    pos=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
                    quat=f"{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}",
                )
                xyz = -c * _T0[0:3, 3] + mujoco_joint["offset"]

                ElementTree.SubElement(
                    body,
                    "joint",
                    name=f"dummy_joint_{i}_{node}",
                    type=mujoco_joint["type"],
                    axis=f"{axis[0]} {axis[1]} {axis[2]}",
                    pos=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
                    range=f"{joint_range[0]} {joint_range[1]}",
                    damping=mujoco_joint["damping"],
                    armature=mujoco_joint["armature"],
                    frictionloss=mujoco_joint["frictionloss"],
                )

                geom_attributes = {
                    "name": f"dummy_geom_{i}_{node}",
                    "type": "sphere",
                    "size": f"{0.01}",
                    "class": "visual",
                    "rgba": f"{0.0} {0.0} {0.0} {0.0}",
                }
                ElementTree.SubElement(body, "geom", geom_attributes)

                parent_xml = body
            else:
                body = ElementTree.SubElement(
                    parent_xml,
                    "body",
                    name=f"body_{node}",
                    pos=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
                    quat=f"{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}",
                )
                xyz = -c * _T0[0:3, 3] + mujoco_joint["offset"]
                ElementTree.SubElement(
                    body,
                    "joint",
                    name=f"joint_{parent_node}_{node}",
                    type=mujoco_joint["type"],
                    axis=f"{axis[0]} {axis[1]} {axis[2]}",
                    pos=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
                    range=f"{joint_range[0]} {joint_range[1]}",
                    damping=mujoco_joint["damping"],
                    armature=mujoco_joint["armature"],
                    frictionloss=mujoco_joint["frictionloss"],
                )

    # Show mesh
    geom_attributes = {
        "name": f"geom_{node}_vis",
        "mesh": f"mesh_{node}",
        "quat": f"{quat_id[0]:.5f} {quat_id[1]:.5f} {quat_id[2]:.5f} {quat_id[3]:.5f}",
        "class": "visual",
        "rgba": f"{color[0]} {color[1]} {color[2]} {1}",
    }
    ElementTree.SubElement(body, "geom", geom_attributes)
    geom_attributes = {
        "name": f"geom_{node}_col",
        "mesh": f"mesh_{node}",
        "quat": f"{quat_id[0]:.5f} {quat_id[1]:.5f} {quat_id[2]:.5f} {quat_id[3]:.5f}",
        "class": "collision",
    }
    ElementTree.SubElement(body, "geom", geom_attributes)

    # Show OBB
    geom_attributes = {
        "name": f"geom_{node}_obb",
        "type": "box",
        "size": f"{bbox[0]:.5f} {bbox[1]:.5f} {bbox[2]:.5f}",
        "quat": f"{quat_id[0]:.5f} {quat_id[1]:.5f} {quat_id[2]:.5f} {quat_id[3]:.5f}",
        "class": "visual",
        "rgba": f"{0.0} {1.0} {0.0} {0.0}",  # {0.0} {1.0} {0.0} {0.25}
    }
    ElementTree.SubElement(body, "geom", geom_attributes)

    # Process the children of this node
    for neighbor in graph.neighbors(node):
        if neighbor != node:  # Avoid adding self
            if parent_node == None or neighbor != parent_node:
                # Recursively add the child body
                add_body_G_(graph, neighbor, body, node)


def generate_mujoco_scene_from_G(
    graph_directory,
    output_directory,
    asset_name,
    graph,
    use_graph_mesh=True,
    embedding_model_cats=None,
    embedding_body_types=None,
    mesh_extractor=None,
    model=None,
    vqvae_2d_model=None,
    sample_image_base_path=None,
    white_background=True,
    device=torch.device("cpu"),
) -> None:
    """
    Generates a MuJoCo XML scene file from graph data and additional asset information.

    This function builds a MuJoCo XML scene similar to generate_mujoco_scene_from_VE but based on networkx
    graph data. It creates directories for assets (meshes, images, text), processes each node of the graph
    to extract or generate meshes (using either provided graph meshes or by generating them via a model),
    and sets up visual, collision, and world body properties in the XML scene. The kinematic tree is built
    recursively via the add_body_G_ function.

    Args:
        graph_directory (str): Directory where the graph output (e.g., json file) is saved.
        output_directory (str): Directory where the scene XML file and asset subdirectories will be saved.
        asset_name (str): Name of the asset; used for naming directories and files.
        graph (networkx.Graph or similar): The graph containing node and edge data for the kinematic tree.
        use_graph_mesh (bool, optional): If True, uses meshes stored in the graph data; otherwise, generates meshes.
                                         Defaults to True.
        embedding_model_cats: (optional) Model categories for embedding, if used.
        embedding_body_types: (optional) Body type embeddings, if used.
        mesh_extractor: (optional) A mesh extractor object with a generate_from_sdf method.
        model: (optional) A model used for inference to generate meshes.
        vqvae_2d_model: (optional) A VQVAE model for generating 2D images; cannot be used with sample_image_base_path.
        sample_image_base_path (str, optional): A directory path to sample images from. Cannot be used with vqvae_2d_model.
        white_background (bool, optional): If True, configures the scene for a white background. Defaults to True.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        None
    """
    # Ensure that only one of vqvae_2d_model or sample_image_base_path is provided
    assert not (sample_image_base_path is not None and vqvae_2d_model is not None)

    # Create the output directory if it doesn't exist
    asset_directory = os.path.join(output_directory, asset_name)
    asset_mesh_directory = os.path.join(asset_directory, "meshes")
    asset_image_directory = os.path.join(asset_directory, "images")
    asset_text_directory = os.path.join(asset_directory, "text")
    scene_path = os.path.join(asset_directory, asset_name + "_scene.xml")
    os.makedirs(asset_mesh_directory, exist_ok=True)
    os.makedirs(asset_image_directory, exist_ok=True)
    os.makedirs(asset_text_directory, exist_ok=True)

    # Create the root element of the XML file with the model attribute
    mujoco = ElementTree.Element("mujoco", {"model": "midgard scene"})

    # Specify memory size for the simulation
    ElementTree.SubElement(mujoco, "size", {"memory": "50M"})

    # Define statistics for simulation analysis
    ElementTree.SubElement(mujoco, "statistic", {"center": "0 0 .3", "extent": "1.2"})

    # Add compiler settings to the XML
    ElementTree.SubElement(
        mujoco,
        "compiler",
        {
            "angle": "radian",
            "balanceinertia": "true",
            "autolimits": "true",
            "fusestatic": "true",
            "convexhull": "false",
        },
    )

    # Add simulation options such as timestep, integrator, solver, etc.
    ElementTree.SubElement(
        mujoco,
        "option",
        {
            "timestep": "0.001",
            "integrator": "implicitfast",
            "solver": "Newton",
            "cone": "pyramidal",
        },
    )

    # Set up default properties
    default = ElementTree.SubElement(mujoco, "default")
    ElementTree.SubElement(default, "material", {"specular": "0", "shininess": "0.25"})

    # Define default visual properties for geometries
    visual_default = ElementTree.SubElement(default, "default", {"class": "visual"})
    ElementTree.SubElement(
        visual_default,
        "geom",
        {"group": "2", "type": "mesh", "contype": "0", "conaffinity": "0"},
    )

    # Define default collision properties for geometries
    collision_default = ElementTree.SubElement(
        default, "default", {"class": "collision"}
    )
    ElementTree.SubElement(
        collision_default,
        "geom",
        {
            "group": "3",
            "type": "mesh",
            "priority": "1",
            "solimp": "0.015 0.09 0.01",
            "condim": "6",
        },
    )

    visual = ElementTree.SubElement(mujoco, "visual")
    asset = ElementTree.SubElement(mujoco, "asset")
    if white_background:
        # Configure visual elements like lighting and color
        ElementTree.SubElement(visual, "rgba", {"haze": "1 1 1 0.5"})
        ElementTree.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

        # Add skybox texture
        ElementTree.SubElement(
            asset,
            "texture",
            {
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "1 1 1",
                "rgb2": "0.5 0.5 0.5",
                "width": "512",
                "height": "3072",
            },
        )
        # Add ground plane texture
        ElementTree.SubElement(
            asset,
            "texture",
            {
                "type": "2d",
                "name": "groundplane",
                "builtin": "checker",
                "mark": "edge",
                "rgb1": "1 1 1",
                "rgb2": "1 1 1",
                "markrgb": "1 1 1",
                "width": "300",
                "height": "300",
            },
        )
    else:
        # Configure visual elements like lighting and color
        ElementTree.SubElement(
            visual,
            "headlight",
            {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"},
        )
        ElementTree.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
        ElementTree.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

        # Add skybox texture
        ElementTree.SubElement(
            asset,
            "texture",
            {
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.3 0.5 0.7",
                "rgb2": "0 0 0",
                "width": "512",
                "height": "3072",
            },
        )
        # Add ground plane texture
        ElementTree.SubElement(
            asset,
            "texture",
            {
                "type": "2d",
                "name": "groundplane",
                "builtin": "checker",
                "mark": "edge",
                "rgb1": "0.2 0.3 0.4",
                "rgb2": "0.1 0.2 0.3",
                "markrgb": "0.8 0.8 0.8",
                "width": "300",
                "height": "300",
            },
        )

    # Define material for the ground plane
    ElementTree.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )

    if sample_image_base_path is not None:
        with open(
            os.path.join(
                os.getcwd(),
                "..",
                "dataset",
                "PartNetMobility",
                "metadata",
                "semantic_to_img.json",
            ),
            "r",
        ) as infile:
            sem_to_meshes = json.load(infile)

    # Asset meshes
    image_processor = ImageProcessor(ImgProcCfg())
    for node in graph.nodes(data=True):
        # Extract mesh data
        bbox_in_graph = np.abs(np.array(node[1]["bbox"]))
        # need to scale and rotate bbox before inputting to the model
        bbox = bbox_in_graph.copy()
        # modify bb --> rotate because rotated when creating sdfs
        if FLIP_AXES_CORRECTION:
            bbox_temp = bbox.copy()
            bbox[0] = bbox_temp[2]
            bbox[2] = bbox_temp[0]
        # scale to 0.75 because that's what the model was trained on
        bbox = bbox / np.max(bbox) * 0.75

        mesh_name = "mesh_" + str(node[1]["node_id"])
        image_name = "image_" + str(node[1]["node_id"])
        txt_name = "text_" + str(node[1]["node_id"])
        if use_graph_mesh and "mesh" in node[1].keys():
            mesh_trimesh = node[1]["mesh"]
            # convert the mesh into open3d format
            # Extract vertices and faces from the Trimesh object
            vertices = np.asarray(mesh_trimesh.vertices)
            faces = np.asarray(mesh_trimesh.faces)

            # Ensure faces have the shape (N, 3)
            if len(faces.shape) == 2:
                # Create an Open3D TriangleMesh
                mesh = o3d.geometry.TriangleMesh()

                # Assign vertices and faces to the Open3D mesh
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)

                try:
                    # If the Trimesh object has vertex normals, add them to the Open3D mesh
                    if (
                        mesh_trimesh.vertex_normals is not None
                        and len(mesh_trimesh.vertex_normals) > 0
                    ):
                        mesh.vertex_normals = o3d.utility.Vector3dVector(
                            mesh_trimesh.vertex_normals
                        )
                except:
                    pass

                try:
                    # If the Trimesh object has vertex colors, add them to the Open3D mesh
                    if (
                        mesh_trimesh.visual.vertex_colors is not None
                        and len(mesh_trimesh.visual.vertex_colors) > 0
                    ):
                        mesh.vertex_colors = o3d.utility.Vector3dVector(
                            mesh_trimesh.visual.vertex_colors
                        )
                except:
                    pass

                try:
                    # Optionally, compute vertex normals if they are not already present
                    if not mesh.has_vertex_normals():
                        mesh.compute_vertex_normals()
                except:
                    pass

            else:
                print("Degenerate mesh. Replacing by its BB.")
                mesh = o3d.geometry.TriangleMesh.create_box(
                    width=2 * bbox_in_graph[0],
                    height=2 * bbox_in_graph[1],
                    depth=2 * bbox_in_graph[2],
                )
                mesh = mesh.translate(-mesh.get_center(), relative=True)  # Centering

            input_txt = ""
        elif not use_graph_mesh and (
            "body_id" in node[1].keys() or "asset_id" in node[1].keys()
        ):
            # Start by resolving the mesh semantic labels:
            if "body_id" in node[1].keys() and "asset_id" in node[1].keys():
                semantic_body_label = node[1]["body_id"]
                semantic_asset_label = node[1]["asset_id"]
                input_txt = (
                    "A "
                    + semantic_body_label.lower()
                    + " as a part of a "
                    + semantic_asset_label.lower()
                )
                # Example: "A drawer as component of a cabinet"
            elif "body_id" in node[1].keys() and not ("asset_id" in node[1].keys()):
                semantic_body_label = node[1]["body_id"]
                semantic_asset_label = ""
                input_txt = "A " + semantic_body_label.lower()
                # Example: "A drawer"
            elif not ("body_id" in node[1].keys()) and "asset_id" in node[1].keys():
                semantic_body_label = ""
                semantic_asset_label = node[1]["asset_id"]
                input_txt = "A part of a " + semantic_asset_label.lower()
                # Example: "A part of a cabinet"
            else:
                semantic_body_label = ""
                semantic_asset_label = ""
                input_txt = ""

            bb_model_type = model.config["shape_generator"]["graphmodel"][
                "architecture"
            ]
            if "gat" in bb_model_type:  # mlp, gat_local or gat_global
                # two options for bb_mode: graph_global (whole graph) with indicator or graph_local (tree only with nodes connected to part)
                mode_global_graph = "global" in bb_model_type

                # instead of loading the npz file, I use "graph" and "node" here TODO: graph seems to have other format than npz file!
                # bb_model_inp = get_graph_bb(npz_input, part_id, self.opt.bb_mode, mode_global_graph)
                bb_model_inp = get_graph_bb(graph, node[0], "", mode_global_graph).to(
                    device
                )
            else:
                bb_model_inp = torch.tensor(bbox).unsqueeze(0).to(device)

            if sample_image_base_path is None:
                assert (
                    vqvae_2d_model is not None
                ), "either vqvae_2d_model or sample_image_base_path must be not None"
                
                print("Generating VQVAE_2D image...")
                image_latent = torch.from_numpy(node[1]["2d_latent"]).reshape(-1, 8, 8).unsqueeze(0).to(device)
                with torch.no_grad():
                    image = vqvae_2d_model.generate_image(
                        image_latent
                    )
                cond_image = image.clone()
                print("VQVAE_2D image generated!")

                img = (
                    image_processor.unnormalize(image[0]).detach().cpu().numpy()
                )
                img *= 255.0
                img = np.clip(img, a_min=0, a_max=255)
                img = img.astype(np.uint8)
                img = img.transpose(1, 2, 0)

            else:  # sample image from path
                sampled_image_path = np.random.choice(
                    sem_to_meshes[semantic_asset_label.lower()][
                        semantic_body_label.lower()
                    ]
                )
                img_path = os.path.join(sample_image_base_path, sampled_image_path)
                
                # apply transforms
                cond_image = image_processor(img_path, train=False).unsqueeze(0).to(device)

            # Pack into a dict
            test_data = {
                "sdf": torch.zeros(1, 64, 64, 64).to(device),
                "txt": [input_txt],
                "graph": bb_model_inp,
                "img": cond_image,
            }

            # construct 3D bb for bb prior method
            res = model.config["dataset"]["grid_resolution"]
            min_bound = res // 2 - (bbox * res).astype(int) // 2
            max_bound = res // 2 + (bbox * res).astype(int) // 2

            if (
                model.config["shape_generator"]["denoising_diffusion"].get(
                    "subtract_bb", True
                )
                == True
            ):
                bb_factor = model.config["shape_generator"]["denoising_diffusion"].get(
                    "bb_prior_factor", 0.02
                )
                test_data["bb_3D"] = (
                    create_bbox_3d(
                        min_bound,
                        max_bound,
                        val=0.5 * bb_factor,
                        res=res,
                        as_bool=False,
                    )
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )

            # Generate mesh
            mm_cond_scale = {"uc": 1, "txt": 1.0, "img": 1.0, "graph": 1.0}
            sdf_gen = model.neural_network.mm_inference(
                test_data,
                scale=mm_cond_scale,
            )

            # Save the generation results
            mesh = mesh_extractor.generate_from_sdf(sdf_gen)
            mesh = mesh.translate(-mesh.get_center(), relative=True)  # Centering
            mesh_vertices = np.asarray(mesh.vertices)

            if FLIP_AXES_CORRECTION:
                # flip the resulting vertices axes before scaling
                verts_temp = mesh_vertices.copy()
                mesh_vertices[:, 0] = verts_temp[:, 2]
                mesh_vertices[:, 2] = verts_temp[:, 0]
                # flip faces to avoid inside out
                mesh_faces_scaled = np.asarray(mesh.triangles)
                mesh_faces_temp = mesh_faces_scaled.copy()
                mesh_faces_scaled[:, 0] = mesh_faces_temp[:, 2]
                mesh_faces_scaled[:, 2] = mesh_faces_temp[:, 0]
                mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_scaled)

            # axis-wise scaling
            desired_pose = {"size": 2 * bbox_in_graph, "pos": np.zeros(3)}
            mesh_vertices_scaled = scale_mesh(mesh_vertices, desired_pose)
            mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_scaled)

        else:
            # Use a cube mesh the size of the bounding box
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=2 * bbox_in_graph[0],
                height=2 * bbox_in_graph[1],
                depth=2 * bbox_in_graph[2],
            )
            mesh = mesh.translate(-mesh.get_center(), relative=True)  # Centering
            input_txt = ""

        # Assign the color to all vertices of the mesh
        color = node[1]["color"]
        # mesh.vertex_colors = o3d.utility.Vector3dVector(
        #     np.tile(color, (len(mesh.vertices), 1))
        # )

        # Perform mesh processing operations
        mesh = (
            mesh.compute_triangle_normals()
            .compute_vertex_normals()
            .simplify_quadric_decimation(target_number_of_triangles=10000)
            .filter_smooth_simple(number_of_iterations=1)
            .remove_degenerate_triangles()
            .remove_duplicated_triangles()
            .remove_duplicated_vertices()
            .remove_non_manifold_edges()
            .remove_unreferenced_vertices()
        )

        # Save to a file
        mesh_path = os.path.join(asset_mesh_directory, mesh_name + ".obj")
        image_path = os.path.join(asset_image_directory, image_name + ".png")
        txt_path = os.path.join(asset_text_directory, txt_name + ".txt")

        try:
            with open(txt_path, "w") as file:
                file.write(input_txt)  #
        except:
            pass

        try:
            imageio.imsave(image_path, img)
        except:
            pass

        o3d.io.write_triangle_mesh(mesh_path, mesh, write_vertex_normals=True)

        # Update nx graph mesh for the ablation
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        trimesh_bb_mesh = trimesh.primitives.Box(extents=2 * abs(bbox), mutable=True)
        node[1]["mesh"] = trimesh_mesh
        node[1]["bb_mesh"] = trimesh_bb_mesh
        fn = os.path.join(graph_directory, asset_name + ".json")
        write_json(nx.adjacency_data(graph), fn)

        # Store mesh path in the xml file
        ElementTree.SubElement(
            asset,
            "mesh",
            {
                # "class": "midgard",
                "file": f"meshes/{mesh_name}.obj",
            },
        )

    # Set up the world body, including lighting and the ground plane
    worldbody = ElementTree.SubElement(mujoco, "worldbody")
    # Light
    ElementTree.SubElement(
        worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"}
    )

    # Floor
    ElementTree.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "size": "0 0 0.05",
            "type": "plane",
            "material": "groundplane",
        },
    )

    # Process and add body elements to the XML
    # Assuming the root of your kinematic tree is named 'root'
    nodes = [node for node in graph.nodes]
    nodes_bbox = np.stack([d["bbox"] for nid, d in graph.nodes(data=True)], 0)
    nodes_volume = nodes_bbox.prod(axis=-1) * 8
    root_node = nodes[nodes_volume.argmax()]
    add_body_G_(graph, root_node, worldbody)

    # Write the XML tree to a file
    pretty_xml = prettify_xml_(mujoco)
    with open(scene_path, "w") as f:
        f.write(pretty_xml)


def prettify_xml_(elem) -> str:
    """
    Returns a pretty-printed XML string for the given Element.

    This function converts an ElementTree Element into a well-formatted, indented XML string,
    making it easier to read. It utilizes defusedxml.minidom to perform the formatting,
    providing a safer way to parse XML.

    Args:
        elem (xml.etree.ElementTree.Element): The XML element to format.

    Returns:
        str: A pretty-printed XML string.
    """
    rough_string = ElementTree.tostring(elem, "utf-8")
    reparsed = parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")