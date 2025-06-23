import os
import platform

if "Darwin" in platform.uname().version:
    pass
elif "microsoft-standard" in platform.uname().release:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
else:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pyrender
from shapely.geometry import Polygon
import sys
from termcolor import cprint
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2mat
import trimesh

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BBOX_CORNER = np.array(
    [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0],
    ]
)


def render(
    # draw pointclouds
    pcl_list=None,
    pcl_color_list=None,
    pcl_radius_list=None,
    pcl_fallback_colors=[0.9, 0.9, 0.9, 1.0],
    # draw meshes
    mesh_list=None,
    mesh_color_list=None,
    # draw lines
    line_list=None,
    lines_color_list=None,
    lines_color_fallback=[0.5, 0.5, 0.5, 1.0],
    # draw arrows
    arrow_tuples=None,  # (Nx3, Nx3)
    arrow_radius=0.01,
    arrow_colors=None,  # N,4
    arrow_head=False,
    # render config
    shape=(640, 640),
    light_intensity=1.0,
    light_vertical_angle=-np.pi / 4,
    yfov=np.pi / 3.0,
    cam_angle_yaw=0.0,
    cam_angle_pitch=0.0,
    cam_angle_z=0.0,
    cam_dist=1.0,
    cam_height=0.0,
    perpoint_color_flag=False,
    render_flags=pyrender.RenderFlags.NONE,
    **kargs,
):
    # Sanity checks
    if not isinstance(cam_angle_yaw, list):
        cam_angle_yaw = [cam_angle_yaw]
    if not isinstance(cam_angle_pitch, list):
        cam_angle_pitch = [cam_angle_pitch]
    if not isinstance(cam_angle_z, list):
        cam_angle_z = [cam_angle_z]
    if not isinstance(cam_dist, list):
        cam_dist = [cam_dist]
    if not isinstance(cam_height, list):
        cam_height = [cam_height]

    N_cam_pose = len(cam_angle_yaw)
    assert len(cam_angle_pitch) == N_cam_pose
    assert len(cam_angle_z) == N_cam_pose
    assert len(cam_dist) == N_cam_pose
    assert len(cam_height) == N_cam_pose

    cam_pose_list = []
    for i in range(N_cam_pose):
        cam_pose = np.eye(4)
        R = euler2mat(cam_angle_yaw[i], -cam_angle_pitch[i], cam_angle_z[i], "ryxz")
        cam_pose[:3, :3] = R
        cam_pose[2, 3] += cam_dist[i]
        cam_pose[:3, 3:] = R @ cam_pose[:3, 3:]
        cam_pose[1, 3] += cam_height[i]
        cam_pose_list.append(cam_pose)

    renderer = pyrender.OffscreenRenderer(shape[0], shape[1])
    scene = pyrender.Scene()

    ######################################################################
    # dlight = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
    # T_dl = np.eye(4)
    # T_dl[:3, :3] = euler2mat(-np.pi / 2.0, 0.0, 0.0, "rxyz")
    # scene.add(dlight, pose=T_dl)

    # plight = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
    # T_pl = np.eye(4)
    # T_pl[1, 3] += light_y
    # scene.add(plight, pose=T_pl)
    ######################################################################

    # light from 4 direction above
    for rot in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        dlight = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0], intensity=light_intensity
        )
        T_dl = np.eye(4)
        T_dl[:3, :3] = euler2mat(light_vertical_angle, rot, 0.0, "sxyz")
        scene.add(dlight, pose=T_dl)

    if pcl_list is not None:
        # PCL [N,3]
        assert len(pcl_list) == len(pcl_radius_list)
        for i in range(len(pcl_list)):
            pcl = pcl_list[i]
            radius = pcl_radius_list[i]
            assert isinstance(radius, float)
            color = pcl_fallback_colors if pcl_color_list is None else pcl_color_list[i]
            if isinstance(color, torch.Tensor):
                color = color.numpy()
            if isinstance(color, np.ndarray) and color.ndim == 1:
                # color = cm.viridis(color)
                # color = cm.seismic(color)
                color = cm.cool(color)
            if radius <= 0.0:
                m = pyrender.Mesh.from_points(pcl, colors=color)
                scene.add(m)
            else:
                sm = trimesh.creation.uv_sphere(radius=radius)
                if perpoint_color_flag and isinstance(color, np.ndarray):
                    for i in tqdm(range(len(pcl))):
                        sm = trimesh.creation.uv_sphere(radius=radius)
                        sm.visual.vertex_colors = color[i]
                        tfs = np.eye(4)
                        tfs[:3, 3] = pcl[i]
                        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                        scene.add(m)
                else:
                    sm.visual.vertex_colors = color
                    tfs = np.tile(np.eye(4), (pcl.shape[0], 1, 1))
                    tfs[:, :3, 3] = pcl
                    m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                    scene.add(m)
    if mesh_list is not None:
        for i in range(len(mesh_list)):
            if len(mesh_list[i].vertices) == 0:
                continue
            if mesh_color_list is not None and mesh_color_list[i] is not None:
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    roughnessFactor=0.0,
                    alphaMode="BLEND",
                    baseColorFactor=mesh_color_list[i],
                )
            else:
                material = None
            scene.add(pyrender.Mesh.from_trimesh(mesh_list[i], material=material))
    if line_list is not None:
        if lines_color_list is not None:
            assert len(lines_color_list) == len(line_list)
        else:
            lines_color_list = [lines_color_fallback] * len(line_list)
        for color, start_end in zip(lines_color_list, line_list):
            lines = np.hstack(start_end)
            lines = lines.reshape(-1, 3)
            line_color = np.asarray(color)
            if line_color.ndim == 1:
                line_color = np.tile(line_color, (len(lines), 1))
            else:
                assert len(line_color) * 2 == len(lines)
                line_color = np.hstack((line_color, line_color)).reshape(-1, 4)
            primitive = [pyrender.Primitive(lines, mode=1, color_0=line_color)]
            primitive_mesh = pyrender.Mesh(primitive)
            scene.add(primitive_mesh)
    if arrow_tuples is not None:
        # ! Note that this does not support list like the line
        arrow_start, arrow_dir = arrow_tuples
        assert len(arrow_start) == len(arrow_colors)
        for aid in range(len(arrow_start)):
            plot_arrow(
                scene,
                arrow_start[aid],
                arrow_dir[aid],
                tube_radius=arrow_radius,
                color=arrow_colors[aid],
                smooth=True,
                use_head=arrow_head,
            )
    rgb_list = []
    for cam_pose in cam_pose_list:
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=shape[0] / shape[1])
        camera = scene.add(camera, pose=cam_pose)
        rgb, _ = renderer.render(scene, flags=render_flags)
        scene.remove_node(camera)
        rgb_list.append(rgb)
    rgb_list = np.concatenate(rgb_list, 1)
    return rgb_list


def plot_arrow(
    scene,
    start_point,
    direction,
    tube_radius=0.01,
    color=(0.5, 0.5, 0.5),
    material=None,
    smooth=True,
    use_head=True,
):
    """Plot an arrow with start and end points.
    Parameters
    ----------
    start_point : (3,) float
        Origin point for the arrow
    direction : (3,) float
        Vector defining the arrow
    tube_radius : float
        Radius of plotted x,y,z axes.
    color : (3,) float
        The color of the tube.
    material:
        Material of mesh
    n_components : int
        The number of edges in each polygon representing the tube.
    smooth : bool
        If true, the mesh is smoothed before rendering.
    """
    end_point = start_point + direction
    if use_head:
        arrow_head = create_arrow_head(
            length=np.linalg.norm(direction), tube_radius=tube_radius
        )
        arrow_head_rot = trimesh.geometry.align_vectors(np.array([0, 0, 1]), direction)
        arrow_head_tf = np.matmul(
            trimesh.transformations.translation_matrix(end_point), arrow_head_rot
        )

    if np.linalg.norm(end_point - start_point, 2) < 1e-6:
        end_point += np.ones_like(end_point) * 1e-4
    vec = np.array([start_point, end_point])

    plot3d_tube(scene, vec, tube_radius=tube_radius, color=color)
    if use_head:
        add_mesh(
            scene,
            arrow_head,
            T_mesh_world=arrow_head_tf,
            color=color,
            material=material,
            smooth=smooth,
        )


def add_mesh(
    scene,
    mesh,
    name=None,
    T_mesh_world=None,
    style="surface",
    color=(0.5, 0.5, 0.5),
    material=None,
    smooth=False,
):
    """Visualize a 3D triangular mesh.
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to visualize.
    name : str
        A name for the object to be added.
    T_mesh_world : autolab_core.RigidTransform
        The pose of the mesh, specified as a transformation from mesh frame to world frame.
    style : str
        Triangular mesh style, either 'surface' or 'wireframe'.
    color : 3-tuple
        Color tuple.
    material:
        Material of mesh
    smooth : bool
        If true, the mesh is smoothed before rendering.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Must provide a trimesh.Trimesh object")

    n = _create_node_from_mesh(
        mesh,
        name=name,
        pose=T_mesh_world,
        color=color,
        material=material,
        poses=None,
        wireframe=(style == "wireframe"),
        smooth=smooth,
    )
    scene.add_node(n)
    return n


def _create_node_from_mesh(
    mesh,
    name=None,
    pose=None,
    color=None,
    material=None,
    poses=None,
    wireframe=False,
    smooth=True,
):
    """Helper method that creates a pyrender.Node from a trimesh.Trimesh"""
    # Create default pose
    if pose is None:
        pose = np.eye(4)

    # Create vertex colors if needed
    if color is not None:
        color = np.asanyarray(color, dtype=np.float32)
        if color.ndim == 1 or len(color) != len(mesh.vertices):
            color = np.repeat(color[np.newaxis, :], len(mesh.vertices), axis=0)
        mesh.visual.vertex_colors = color

    if material is None and mesh.visual.kind != "texture":
        if color is not None:
            material = None
        else:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
                metallicFactor=0.2,
                roughnessFactor=0.8,
            )

    m = pyrender.Mesh.from_trimesh(
        mesh, material=material, poses=poses, wireframe=wireframe, smooth=smooth
    )
    return pyrender.Node(mesh=m, name=name, matrix=pose)


import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def plot3d_tube(
    scene,
    points,
    tube_radius=None,
    name=None,
    pose=None,
    color=(0.5, 0.5, 0.5),
    material=None,
    n_components=16,
    smooth=True,
):
    """Plot a 3d curve through a set of points using tubes.
    Parameters
    ----------
    points : (n,3) float
        A series of 3D points that define a curve in space.
    tube_radius : float
        Radius of tube representing curve.
    name : str
        A name for the object to be added.
    pose : autolab_core.RigidTransform
        Pose of object relative to world.
    color : (3,) float
        The color of the tube.
    material:
        Material of mesh
    n_components : int
        The number of edges in each polygon representing the tube.
    smooth : bool
        If true, the mesh is smoothed before rendering.
    """
    # Generate circular polygon
    vec = np.array([0.0, 1.0]) * tube_radius
    angle = 2 * np.pi / n_components
    rotmat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    perim = []
    for _ in range(n_components):
        perim.append(vec)
        vec = np.dot(rotmat, vec)
    poly = Polygon(np.stack(perim, 0))

    # Sweep it along the path
    mesh = trimesh.creation.sweep_polygon(poly, points)
    return add_mesh(
        scene,
        mesh,
        name=name,
        T_mesh_world=pose,
        color=color,
        material=material,
        smooth=smooth,
    )


def create_arrow_head(length=0.1, tube_radius=0.005, n_components=30):
    """from https://github.com/BerkeleyAutomation/visualization"""
    radius = tube_radius * 2.0
    height = length * 0.5

    # create a 2D pie out of wedges
    theta = np.linspace(0, np.pi * 2, n_components)
    vertices = (
        np.column_stack((np.sin(theta), np.cos(theta), np.zeros(len(theta)))) * radius
    )

    # the single vertex at the center of the circle
    # we're overwriting the duplicated start/end vertex
    # plus add vertex at tip of cone
    vertices[0] = [0, 0, 0]
    vertices = np.append(vertices, [[0, 0, height]], axis=0)

    # whangle indexes into a triangulation of the pie wedges
    index = np.arange(1, len(vertices)).reshape((-1, 1))
    index[-1] = 1
    faces_2d = np.tile(index, (1, 2)).reshape(-1)[1:-1].reshape((-1, 2))
    faces = np.column_stack((np.zeros(len(faces_2d), dtype=np.int), faces_2d))

    # add triangles connecting to vertex above
    faces = np.append(
        faces,
        np.column_stack(
            ((len(faces_2d) + 1) * np.ones(len(faces_2d), dtype=np.int), faces_2d)
        )[:, ::-1],
        axis=0,
    )

    arrow_head = trimesh.Trimesh(faces=faces, vertices=vertices)
    return arrow_head


# if __name__ == "__main__":
#     pcl = np.random.rand(100, 3)
#     render(pcl_list=[pcl], pcl_color_list=[(1, 0, 0)], pcl_radius_list=[0.01])


def append_mesh_to_G(G, mesh_list, key="mesh"):
    for v, v_data in G.nodes(data=True):
        if mesh_list[v] is None:
            continue

        # Apply SVD to the rotation component
        U, _, Vt = np.linalg.svd(v_data["mesh_RT"], full_matrices=True)

        # Reconstruct the rotation matrix with a determinant of 1
        R_adj = np.dot(U, Vt)

        # Check if the adjustment is necessary
        if np.linalg.det(R_adj) < 0:
            U[
                :, -1
            ] *= -1  # Flip the sign of the last column of U if determinant is negative
            R_adj = np.dot(U, Vt)  # Recompute R_adj to ensure det(R_adj) is positive
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R_adj
        mesh = mesh_list[v]
        mesh.apply_transform(transformation_matrix)
        nx.set_node_attributes(G, {v: {key: mesh}})
    return G


def viz_G_topology(
    G,
    r_edge_color="tab:red",
    p_edge_color="tab:blue",
    hybrid_edge_color="tab:orange",
    node_size=800,
    title="",
    fig_size=(3, 3),
    dpi=100,
    r_range_th=3e-3,
    p_range_th=3e-3,
    show_border=True,
    use_categorical_asset=False,
    use_categorical_body=False,
    use_categorical_joint=False,
):
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    pos = nx.kamada_kawai_layout(G)
    options = {"edgecolors": "tab:gray", "node_size": node_size, "alpha": 0.9}
    labels = {}
    asset_id = ""
    for n, n_data in G.nodes(data=True):
        nx.draw_networkx_nodes(
            G, pos, nodelist=[n], node_color=G.nodes[n]["color"], **options
        )
        if use_categorical_body:
            if "body_id" in n_data:
                labels[n] = n_data["body_id"]
            else:
                labels[n] = n

        if use_categorical_asset:
            if "asset_id" in n_data:
                asset_id = n_data["asset_id"]
            else:
                asset_id = "Unknown"

        # print("viz_utils.py: ", n_data["body_id"], " | ", n_data["asset_id"])

    nx.draw_networkx_labels(G, pos, labels, font_size=12, horizontalalignment="right")
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=1.0)
    for e0, e1, e_data in G.edges(data=True):
        r_lim, p_lim = np.array(e_data["rlim"]), np.array(e_data["plim"])
        r_range = abs(r_lim[1] - r_lim[0])
        p_range = abs(p_lim[1] - p_lim[0])
        if use_categorical_joint:
            label = e_data["joint_label"].argmax()
            if label == 0:  # Screw
                edge_color = hybrid_edge_color
            elif label == 1:  # Revolute
                edge_color = r_edge_color
            else:  # Prismatic
                edge_color = p_edge_color
        else:
            if r_range > r_range_th and p_range > p_range_th:
                edge_color = hybrid_edge_color
            elif r_range > r_range_th:
                edge_color = r_edge_color
            else:
                edge_color = p_edge_color

        nx.draw_networkx_edges(
            G, pos, edgelist=[(e0, e1)], width=8, alpha=0.7, edge_color=edge_color
        )
    edge_colors = [r_edge_color, p_edge_color, hybrid_edge_color]
    edge_types = ["Revolute", "Prismatic", "Screw"]
    legend_handles = [
        Line2D([0], [0], color=color, lw=4, label=f"{edge_types[i]}")
        for i, color in enumerate(edge_colors)
    ]
    plt.legend(handles=legend_handles, loc="lower right", title="Joint type")
    if use_categorical_asset:
        plt.title(f"{asset_id}_{title}: {len(G.nodes)} bodies, {len(G.edges)} joints")
    else:
        plt.title(f"{title}: {len(G.nodes)} bodies, {len(G.edges)} joints")

    if not show_border:
        plt.axis("off")
    ax = plt.gca()
    fig.tight_layout(pad=1.0)
    rgb = plot_to_image(fig)
    plt.close(fig)
    return rgb


def resolve_range(joint_label, rlim, plim):
    label = joint_label.argmax()
    if label == 0:  # Screw
        pass
    elif label == 1:  # Revolute
        plim *= 0
    else:  # Prismatic
        rlim *= 0

    return rlim, plim


def screw_to_T(theta, d, l, m):
    """
    Computes a transformation matrix from screw theory parameters.

    This function calculates a transformation matrix based on the given screw parameters.
    A screw is defined by its direction, location, pitch, and magnitude of the rotation or translation.

    Args:
    theta (float): The angle of rotation about the screw axis (in radians).
    d (float): The distance of translation along the screw axis.
    l (np.array): The unit vector representing the direction of the screw axis.
    m (np.array): The moment vector of the screw, representing the axis location.

    Returns:
    np.array: A 4x4 transformation matrix representing the screw transformation.

    The transformation matrix T is computed as follows:
    - The rotation part (R) is calculated from the axis-angle representation (l, theta).
    - The translation part (t) is calculated using the formula:
      t = (I - R) * (l x m) + l * d, where 'x' denotes the cross product.
    - The rotation R and translation t are then composed into the 4x4 matrix T.
    """
    # Ensure that l is a unit vector
    assert abs(np.linalg.norm(l) - 1.0) < 1e-4, "The vector is not normalized"

    # Calculate the rotation matrix R from the axis-angle representation
    R = axangle2mat(l, theta)

    # Compute the translation vector t
    t = (np.eye(3) - R) @ (np.cross(l, m)) + l * d

    # Initialize a 4x4 identity matrix
    T = np.eye(4)

    # Populate the rotation part
    T[:3, :3] = R

    # Populate the translation part
    T[:3, 3] = t

    return T


def viz_G_BB(
    G,
    bbox_thickness=0.03,
    bbox_alpha=0.6,
    viz_frame_N=16,
    cam_dist=4.0,
    pitch=np.pi / 4.0,
    yaw=np.pi / 4.0,
    shape=(480, 480),
    light_intensity=1.0,
    light_vertical_angle=-np.pi / 3.0,
    cat_dim=1,
    moving_mask=None,
    mesh_key="mesh",
    viz_box=True,
    render_flags=0,
    use_categorical_joint=False,
):
    """
    Graph visualization routine
    """
    ret = []
    # Check Graph Validity: Verifies that the graph has at least two nodes and is a tree (connected and acyclic).
    if len(G.nodes) >= 2 and nx.is_tree(G):  # now only support tree viz
        # Determines the root node of the tree based on the volume of the bounding boxes.
        node_id = [n for n in G.nodes]
        v_bbox = np.stack([d["bbox"] for nid, d in G.nodes(data=True)], 0)
        v_volume = v_bbox.prod(axis=-1) * 8
        root_vid = node_id[v_volume.argmax()]

        # Sample a set of possible angles for each joint to simulate motion in the visualization.
        for step in range(viz_frame_N):
            node_traverse_list = [n for n in nx.dfs_preorder_nodes(G, root_vid)]
            T_rl_list = [np.eye(4)]  # p_root = T_rl @ p_link
            # * prepare the node pos
            for i in range(len(node_traverse_list) - 1):
                # ! find the parent!
                cid = node_traverse_list[i + 1]

                for e, e_data in G.edges.items():
                    if cid in e:
                        # determine the direction by ensure the other end is a predessor in the traversal list
                        other_end = e[0] if e[1] == cid else e[1]
                        if node_traverse_list.index(other_end) > i:
                            continue
                        else:
                            pid = other_end

                        # Compute transformation matrices for each node that represent
                        # its position and orientation relative to the root.
                        # T1: e_T_src_j1, T2: e_T_j2_dst
                        e_data = G.edges[e]
                        _T0 = np.array(e_data["T_src_dst"])
                        plucker = np.array(e_data["plucker"])  # local plucker
                        l, m = plucker[:3], plucker[3:]
                        plim, rlim = np.array(e_data["plim"]), np.array(e_data["rlim"])
                        if use_categorical_joint:
                            joint_label = e_data["joint_label"]
                            rlim, plim = resolve_range(joint_label, rlim, plim)
                        if moving_mask is None or moving_mask[e]:
                            theta = np.linspace(*rlim, viz_frame_N)[step]
                            d = np.linspace(*plim, viz_frame_N)[step]
                        else:  # don't move
                            theta = np.linspace(*rlim, viz_frame_N)[0]
                            d = np.linspace(*plim, viz_frame_N)[0]
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

                        # Extract the rotation component (R) and translation vector (t)
                        R = T_root_child[:3, :3]
                        t = T_root_child[:3, 3]

                        # Apply SVD to the rotation component
                        U, _, Vt = np.linalg.svd(R, full_matrices=True)

                        # Reconstruct the rotation matrix with a determinant of 1
                        R_adj = np.dot(U, Vt)

                        # Check if the adjustment is necessary
                        if np.linalg.det(R_adj) < 0:
                            U[
                                :, -1
                            ] *= (
                                -1
                            )  # Flip the sign of the last column of U if determinant is negative
                            R_adj = np.dot(
                                U, Vt
                            )  # Recompute R_adj to ensure det(R_adj) is positive

                        # Reassemble the transformation matrix with the adjusted rotation part
                        T_adj = np.eye(4)  # Start with an identity matrix
                        T_adj[:3, :3] = R_adj
                        T_adj[:3, 3] = t

                        T_rl_list.append(T_adj)
                        break
            assert len(T_rl_list) == len(node_traverse_list)

            # * prepare the bbox
            bbox_edge_start_list, bbox_edge_dir_list = [], []
            bbox_corner_list, bbox_color_list = [], []
            pcl_color_list = []
            # bbox_colors = cm.hsv(np.linspace(0, 1, len(node_traverse_list) + 1))[:-1]

            mesh_list, mesh_color_list = [], []
            for nid, T in zip(node_traverse_list, T_rl_list):
                bbox = G.nodes[nid]["bbox"]
                color = G.nodes[nid]["color"]
                pcl_color_list.append(color)
                mesh = trimesh.primitives.Box(extents=2 * bbox, mutable=True)
                mesh.apply_transform(T.copy())
                mesh_list.append(mesh)
                mesh_color_list.append([c * 0.5 for c in color[:3]] + [0.7])
                bbox_corner = BBOX_CORNER * bbox
                bbox_corner = bbox_corner
                bbox_corner = bbox_corner @ T[:3, :3].T + T[:3, 3]
                bbox_corner_list.append(bbox_corner)
                bbox_edge_start_ind = [0, 0, 2, 1, 3, 3, 5, 6, 0, 1, 4, 2]
                bbox_edge_end_ind = [1, 2, 4, 4, 6, 5, 7, 7, 3, 6, 7, 5]
                bbox_start = bbox_corner[bbox_edge_start_ind]
                bbox_end = bbox_corner[bbox_edge_end_ind]
                bbox_edge_start_list.append(bbox_start)
                bbox_edge_dir_list.append(bbox_end - bbox_start)
                bbox_color = color
                bbox_color[-1] = bbox_alpha
                bbox_color_list.append(np.tile(bbox_color[None, :], [12, 1]))

            if len(bbox_corner_list) > 0:
                bbox_color_list = np.concatenate(bbox_color_list, 0)
                bbox_edge_start_list = np.concatenate(bbox_edge_start_list, 0)
                bbox_edge_dir_list = np.concatenate(bbox_edge_dir_list, 0)
                arrow_tuples = (bbox_edge_start_list, bbox_edge_dir_list)
            else:
                bbox_color_list, bbox_edge_start_list, bbox_edge_dir_list = (
                    None,
                    None,
                    None,
                )
                arrow_tuples = None

            # Call a rendering function to create images of the current state of the graph from the camera's perspective.
            rgb0 = render(
                mesh_list=mesh_list,
                mesh_color_list=mesh_color_list,
                cam_dist=cam_dist,
                cam_angle_pitch=pitch,
                cam_angle_yaw=yaw,
                shape=shape,
                light_intensity=light_intensity,
                light_vertical_angle=light_vertical_angle,
                render_flags=render_flags,
            )
            rgb1 = render(
                pcl_list=bbox_corner_list,
                pcl_color_list=pcl_color_list,
                pcl_radius_list=[bbox_thickness * 2.0] * len(bbox_corner_list),
                # arrows
                arrow_head=False,
                arrow_tuples=arrow_tuples,
                arrow_colors=bbox_color_list,
                arrow_radius=bbox_thickness,
                cam_dist=cam_dist,
                cam_angle_pitch=pitch,
                cam_angle_yaw=yaw,
                shape=shape,
                light_intensity=light_intensity,
                light_vertical_angle=light_vertical_angle,
                render_flags=render_flags,
            )
            rgb = np.concatenate([rgb0, rgb1], cat_dim)
            ret.append(rgb)
    else:
        dummy = np.ones((shape[0], shape[1] * 2, 3), dtype=np.uint8) * 127
        ret = [dummy] * viz_frame_N
    ret = ret + ret[::-1]
    return ret


def viz_G(
    G,
    bbox_thickness=0.03,
    bbox_alpha=0.6,
    viz_frame_N=16,
    cam_dist=4.0,
    pitch=np.pi / 4.0,
    yaw=np.pi / 4.0,
    shape=(480, 480),
    light_intensity=1.0,
    light_vertical_angle=-np.pi / 3.0,
    cat_dim=1,
    moving_mask=None,
    mesh_key="mesh",
    viz_box=True,
    render_flags=0,
    use_categorical_joint=False,
):
    """
    Graph visualization routine
    """
    ret = []
    if len(G.nodes) >= 2 and nx.is_tree(G):  # now only support tree viz
        # * now G is connected and acyclic
        # find the root
        vid = [n for n in G.nodes]
        v_bbox = np.stack([d["bbox"] for nid, d in G.nodes(data=True)], 0)
        v_volume = v_bbox.prod(axis=-1) * 8
        root_vid = vid[v_volume.argmax()]

        # * sample a set of possible angle range for each joint
        for step in range(viz_frame_N):
            node_traverse_list = [n for n in nx.dfs_preorder_nodes(G, root_vid)]
            T_rl_list = [np.eye(4)]  # p_root = T_rl @ p_link
            # * prepare the node pos
            for i in range(len(node_traverse_list) - 1):
                # ! find the parent!
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
                        _T0 = np.array(e_data["T_src_dst"])
                        plucker = np.array(e_data["plucker"])
                        l, m = plucker[:3], plucker[3:]
                        plim, rlim = np.array(e_data["plim"]), np.array(e_data["rlim"])
                        if use_categorical_joint:
                            joint_label = e_data["joint_label"]
                            rlim, plim = resolve_range(joint_label, rlim, plim)
                        if moving_mask is None or moving_mask[e]:
                            theta = np.linspace(*rlim, viz_frame_N)[step]
                            d = np.linspace(*plim, viz_frame_N)[step]
                        else:  # don't move
                            theta = np.linspace(*rlim, viz_frame_N)[0]
                            d = np.linspace(*plim, viz_frame_N)[0]
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

                        # Extract the rotation component (R) and translation vector (t)
                        R = T_root_child[:3, :3]
                        t = T_root_child[:3, 3]

                        # Apply SVD to the rotation component
                        U, _, Vt = np.linalg.svd(R, full_matrices=True)

                        # Reconstruct the rotation matrix with a determinant of 1
                        R_adj = np.dot(U, Vt)

                        # Check if the adjustment is necessary
                        if np.linalg.det(R_adj) < 0:
                            U[
                                :, -1
                            ] *= (
                                -1
                            )  # Flip the sign of the last column of U if determinant is negative
                            R_adj = np.dot(
                                U, Vt
                            )  # Recompute R_adj to ensure det(R_adj) is positive

                        # Reassemble the transformation matrix with the adjusted rotation part
                        T_adj = np.eye(4)  # Start with an identity matrix
                        T_adj[:3, :3] = R_adj
                        T_adj[:3, 3] = t

                        T_rl_list.append(T_adj)
                        break
            assert len(T_rl_list) == len(node_traverse_list)

            # * prepare the bbox
            bbox_edge_start_list, bbox_edge_dir_list = [], []
            bbox_corner_list, bbox_color_list = [], []
            pcl_color_list = []
            # bbox_colors = cm.hsv(np.linspace(0, 1, len(node_traverse_list) + 1))[:-1]
            mesh_list, mesh_color_list = [], []
            for nid, T in zip(node_traverse_list, T_rl_list):
                bbox = np.array(G.nodes[nid]["bbox"])
                color = np.array(G.nodes[nid]["color"])
                pcl_color_list.append(color)

                if mesh_key in G.nodes[nid].keys():
                    mesh_dict = G.nodes[nid][mesh_key].copy()
                    mesh = trimesh.Trimesh(
                        vertices=np.array(mesh_dict["vertices"]),
                        faces=np.array(mesh_dict["faces"])
                    )
                    mesh.apply_transform(T.copy())
                    mesh_list.append(mesh)
                    mesh_color_list.append([c * 0.5 for c in color[:3]] + [0.7])
                else:
                    mesh = trimesh.primitives.Box(extents=2 * bbox, mutable=True)
                    mesh.apply_transform(T.copy())
                    mesh_list.append(mesh)
                    mesh_color_list.append([c * 0.5 for c in color[:3]] + [0.7])

                bbox_corner = BBOX_CORNER * bbox
                bbox_corner = bbox_corner
                bbox_corner = bbox_corner @ T[:3, :3].T + T[:3, 3]
                bbox_corner_list.append(bbox_corner)
                bbox_edge_start_ind = [0, 0, 2, 1, 3, 3, 5, 6, 0, 1, 4, 2]
                bbox_edge_end_ind = [1, 2, 4, 4, 6, 5, 7, 7, 3, 6, 7, 5]
                bbox_start = bbox_corner[bbox_edge_start_ind]
                bbox_end = bbox_corner[bbox_edge_end_ind]
                bbox_edge_start_list.append(bbox_start)
                bbox_edge_dir_list.append(bbox_end - bbox_start)
                bbox_color = color
                bbox_color[-1] = bbox_alpha
                bbox_color_list.append(np.tile(bbox_color[None, :], [12, 1]))

            if len(bbox_corner_list) > 0:
                bbox_color_list = np.concatenate(bbox_color_list, 0)
                bbox_edge_start_list = np.concatenate(bbox_edge_start_list, 0)
                bbox_edge_dir_list = np.concatenate(bbox_edge_dir_list, 0)
                arrow_tuples = (bbox_edge_start_list, bbox_edge_dir_list)
            else:
                bbox_color_list, bbox_edge_start_list, bbox_edge_dir_list = (
                    None,
                    None,
                    None,
                )
                arrow_tuples = None

            rgb0 = render(
                mesh_list=mesh_list,
                mesh_color_list=mesh_color_list,
                cam_dist=cam_dist,
                cam_angle_pitch=pitch,
                cam_angle_yaw=yaw,
                shape=shape,
                light_intensity=light_intensity,
                light_vertical_angle=light_vertical_angle,
                render_flags=render_flags,
            )
            if viz_box:
                rgb1 = render(
                    pcl_list=bbox_corner_list,
                    pcl_color_list=pcl_color_list,
                    pcl_radius_list=[bbox_thickness * 2.0] * len(bbox_corner_list),
                    # arrows
                    arrow_head=False,
                    arrow_tuples=arrow_tuples,
                    arrow_colors=bbox_color_list,
                    arrow_radius=bbox_thickness,
                    cam_dist=cam_dist,
                    cam_angle_pitch=pitch,
                    cam_angle_yaw=yaw,
                    shape=shape,
                    light_intensity=light_intensity,
                    light_vertical_angle=light_vertical_angle,
                    render_flags=render_flags,
                )
                rgb = np.concatenate([rgb0, rgb1], cat_dim)
            else:
                rgb = rgb0
            ret.append(rgb)
    else:
        dummy = np.ones((shape[0], shape[1] * 2, 3), dtype=np.uint8) * 127
        ret = [dummy] * viz_frame_N
    ret = ret + ret[::-1]
    return ret


def plot_to_image(fig):
    """
    Convert a Matplotlib figure to a 3D NumPy array with RGB channels and return it.

    The 3D array can be used to display the plot in UI frameworks or for further image processing.

    Parameters:
    - fig: A Matplotlib figure object to be converted to an RGB image.

    Returns:
    - A NumPy array of shape (H, W, 3) where H and W are the height and width of the figure.
      The array represents the RGB image of the figure.

    """
    # Render the figure canvas to make sure it's up-to-date.
    fig.canvas.draw()

    # Grab the RGBA buffer from the figure's canvas.
    buf = fig.canvas.buffer_rgba()

    # The width and height of the canvas:
    width, height = fig.canvas.get_width_height()

    # Convert to a (height x width x 4) NumPy array (uint8).
    image = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))

    # Return the RGB image as a NumPy array.
    return image[:, :, :3]