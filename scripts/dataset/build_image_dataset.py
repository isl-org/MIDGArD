import os
import platform

if "Darwin" in platform.uname().version:
    pass
elif "microsoft-standard" in platform.uname().release:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
else:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import json
import numpy as np
import open3d as o3d
from PIL import Image
import sys
from tqdm import tqdm
import trimesh
import pyrender


def rotation_matrix(axis, angle):
    """
    Returns a 3x3 rotation matrix for rotating 'angle' radians around 'axis'.
    """
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    C = 1 - cos_a
    x, y, z = axis
    return np.array(
        [
            [cos_a + x * x * C, x * y * C - z * sin_a, x * z * C + y * sin_a],
            [y * x * C + z * sin_a, cos_a + y * y * C, y * z * C - x * sin_a],
            [z * x * C - y * sin_a, z * y * C + x * sin_a, cos_a + z * z * C],
        ]
    )


def create_homogeneous(R, t=np.zeros(3)):
    """
    Converts a 3x3 rotation matrix and translation vector to a 4x4 homogeneous matrix.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def create_camera_pose(eye, target, up):
    """
    Returns a 4x4 camera pose matrix (world transformation) given the camera 'eye' position,
    a 'target' point, and an 'up' direction.

    The camera will be oriented such that it looks toward the target and its up direction
    is aligned with the provided 'up' vector.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # In many graphics frameworks (including pyrender), the camera looks along -Z.
    rot = np.column_stack((right, true_up, -forward))
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = eye
    return pose


def render_mesh_pyrender(mesh_path, viewpoints_num, height, width, renderer):
    """
    Renders a 3D mesh from multiple viewpoints using pyrender and saves the resulting images
    with a transparent background.

    The function:
      - Loads a 3D mesh from a supported file (e.g., .ply, .stl, .obj, .off, etc.) using trimesh.
      - Sets up a pyrender scene with an offscreen camera and a directional light.
      - Centers and applies an initial rotation (around the X-axis by pi/6) to the mesh.
      - Rotates the mesh about the Y-axis to generate 'viewpoints_num' distinct views.
      - Renders an image at each view, processes the image to make the bright green background transparent,
        and saves it to an output folder determined from the mesh file's path.

    Args:
        mesh_path (str): Path to the 3D mesh file.
        viewpoints_num (int): Number of distinct viewpoints.
        height (int): Height in pixels of the output images.
        width (int): Width in pixels of the output images.

    Returns:
        None

    Raises:
        Exception: If the mesh file format is not supported.
        AssertionError: If viewpoints_num, height, or width are not greater than 0.
    """
    # Input sanity checks.
    assert viewpoints_num > 0, "Please provide a number of view points greater than 0!"
    assert height > 0, "Please provide a height greater than 0!"
    assert width > 0, "Please provide a width greater than 0!"

    # Check if the file extension is supported.
    supported_exts = (".obj", ".ply", ".stl", ".3ds", ".dae", ".off")
    if not mesh_path.lower().endswith(supported_exts):
        raise Exception(f"Model format not supported: {mesh_path}")

    # Load mesh using trimesh and compute vertex normals if available.
    mesh_trimesh = trimesh.load(mesh_path, force="mesh")
    if not hasattr(mesh_trimesh, "vertex_normals"):
        mesh_trimesh.compute_vertex_normals()

    # Mesh index without extension (ie. 5 if mesh is 38642_5.off)
    idx = os.path.splitext(os.path.basename(mesh_path))[0]

    # Folder that will contain the different images captured from this mesh
    path_parts = mesh_path.split(os.sep)
    path_index = path_parts.index("meshes") - 1
    path = os.sep.join(path_parts[: path_index + 1])
    output_image_folder = os.path.join(path, "images", idx)

    # Create a pyrender mesh from the trimesh object.
    render_mesh_obj = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=False)

    # Create a pyrender scene and set the background to black.
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 1.0]), ambient_light=[0.1, 0.1, 0.1]
    )

    # Center the mesh: translate so that its centroid is at the origin.
    mesh_center = mesh_trimesh.centroid
    T_center = np.eye(4)
    T_center[:3, 3] = -mesh_center

    # Apply an initial rotation around the X-axis by pi/6.
    R_x = rotation_matrix([1, 0, 0], np.pi / 6)
    T_rot_x = create_homogeneous(R_x)
    # Initial transform: first center the mesh, then rotate.
    current_transform = T_rot_x @ T_center

    # Add the mesh to the scene and keep a reference to its node for later updates.
    mesh_node = scene.add(render_mesh_obj, pose=current_transform)

    # Determine a camera position such that the mesh is in view.
    bbox = mesh_trimesh.bounds
    diameter = np.linalg.norm(bbox[1] - bbox[0])
    camera_distance = 10.0 * diameter if diameter > 0 else 10.0
    eye = np.array([0.0, 0.0, camera_distance])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    camera_pose = create_camera_pose(eye, target, up)

    # Create an intrinsics camera using provided parameters.
    # Note: pyrender's IntrinsicsCamera expects fx, fy, cx, cy.
    camera = pyrender.IntrinsicsCamera(
        fx=1000.0, fy=1000.0, cx=width / 2.0, cy=height / 2.0, znear=0.1, zfar=1000.0
    )
    scene.add(camera, pose=camera_pose)

    # Add a directional light at the camera position to illuminate the mesh.
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Precompute the Y-axis rotation matrix for iterative rotations.
    angle_increment = 2 * np.pi / viewpoints_num
    R_y = rotation_matrix([0, 1, 0], angle_increment)
    T_rot_y = create_homogeneous(R_y)

    # For each viewpoint: update the mesh pose, render the scene, post-process, and save.
    for i in range(viewpoints_num):
        # Run the visualization for the current shot
        scene.set_pose(mesh_node, current_transform)
        color, _ = renderer.render(scene)

        # Apply rotation to the mesh
        scene.set_pose(mesh_node, current_transform)

        # Convert the image to RGBA if it's not already
        image = Image.fromarray(color).convert("RGBA")

        # Save the current frame
        rendered_images_path = os.path.join(output_image_folder, str(i) + ".png")
        image.save(rendered_images_path)

        # Rotate the mesh for the next viewpoint.
        current_transform = T_rot_y @ current_transform


def render_mesh_o3d(mesh_path, viewpoints_num, height, width):
    """
    Renders a 3D mesh from multiple viewpoints and saves the resulting images with a transparent background.

    The function loads a 3D mesh from the specified file (supporting formats such as .ply, .stl, .obj, .off, etc.),
    sets up an Open3D visualizer with a static camera, rotates the mesh to generate a specified number of viewpoints,
    captures an image at each viewpoint, processes the image to make a predefined background color transparent, and
    saves each image in an output folder. The output folder is created based on the mesh file's location and name.

    Args:
        mesh_path (str): Path to the 3D mesh file.
        viewpoints_num (int): Number of distinct viewpoints from which the mesh will be rendered.
        height (int): Height (in pixels) of the output images.
        width (int): Width (in pixels) of the output images.

    Returns:
        None

    Raises:
        Exception: If the mesh file format is not supported.
        AssertionError: If any of the input parameters (viewpoints_num, height, or width) are not greater than 0.
    """
    # Sanity checks
    assert viewpoints_num > 0, "Please provide a number of view points greater than 0!"
    assert height > 0, "Please provide a height greater than 0!"
    assert width > 0, "Please provide a width greater than 0!"

    # Check if the file extension is supported.
    supported_exts = (".obj", ".ply", ".stl", ".3ds", ".dae", ".off")
    if not mesh_path.lower().endswith(supported_exts):
        raise Exception(f"Model format not supported: {mesh_path}")

    # Load mesh using Open3D and compute vertex normals if available.
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Mesh index without extension (ie. 5 if mesh is 38642_5.off)
    idx = os.path.splitext(os.path.basename(mesh_path))[0]

    # Folder that will contain the different images captured from this mesh
    path_parts = mesh_path.split(os.sep)
    path_index = path_parts.index("meshes") - 1
    path = os.sep.join(path_parts[: path_index + 1])
    output_image_folder = os.path.join(path, "images", idx)

    # Create an Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=width, height=height, visible=False
    )  # Set the window size to match camera intrinsics

    # Import the geometry into the visualizer
    vis.add_geometry(mesh)

    # Set background to transparent -- this is purely hypothetical
    vis.get_render_option().background_color = np.array([0.0, 0.0, 0.0])

    # Static camera setup
    Fx, Fy = 1000, 1000
    Cx, Cy = width / 2, height / 2  # center of the image
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, Fx, Fy, Cx, Cy)
    intrinsic.intrinsic_matrix = [[Fx, 0, Cx], [0, Fy, Cy], [0, 0, 1]]
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    cam.extrinsic = np.eye(4)
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

    # Take viewpoints_num shots of the same mesh rotated around the origin
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        [np.pi / 6, 0, 0]
    )
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        [0, 2 * np.pi / viewpoints_num, 0]
    )
    for i in range(viewpoints_num):

        # Run the visualization for the current shot
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

        # Apply rotation to the mesh
        mesh.rotate(rotation_matrix, center=mesh.get_center())

        # Save the current frame
        rendered_images_path = os.path.join(output_image_folder, str(i) + ".png")

        # Capture image data to numpy array
        image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

        # Convert the image to RGBA if it's not already
        image = image.convert("RGBA")

        # Save the current frame
        rendered_images_path = os.path.join(output_image_folder, str(i) + ".png")
        image.save(rendered_images_path)

    vis.destroy_window()


def main(argv) -> None:
    """
    Main entry point for the mesh rendering routine.

    This function parses command-line arguments to set up parameters for rendering meshes,
    validates the existence of necessary directories, and loads metadata to build an inverted index
    mapping asset codes to categories. It then copies missing mesh files to the target directory,
    sets up output folders for images, and uses parallel processing to render each mesh from multiple
    viewpoints by calling the render_mesh function.

    Args:
        argv (list of str): Command-line arguments passed to the script.

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
        "--viewpoints_num",
        type=int,
        default=24,
        help="Number of distinct views of the considered body.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=137,
        help="Height of the output images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=137,
        help="Width of the output images.",
    )
    args = parser.parse_args(argv)

    # Sanity checks
    data_directory = os.path.join(args.dataset_directory, "data")
    metadata_directory = os.path.join(args.dataset_directory, "metadata")
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
                continue

            mesh_files_dict[inverted_index[category_code]].append(mesh_path)
            img_dir = os.path.join(
                data_directory,
                inverted_index[category_code],
                category_code,
                "images",
                instance_number,
            )
            os.makedirs(img_dir, exist_ok=True)

    for cat_id in info["all_cats"]:

        # Mesh files for a specific category of asset
        mesh_files = mesh_files_dict[cat_id]
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

        # Create an offscreen renderer.
        renderer = pyrender.OffscreenRenderer(
            viewport_width=args.width, viewport_height=args.height
        )
        for mesh_path in tqdm(mesh_files, desc="Processing meshes"):
            render_mesh_pyrender(
                mesh_path, args.viewpoints_num, args.height, args.width, renderer
            )
        # Clean up the renderer.
        renderer.delete()


# Main execution logic
if __name__ == "__main__":
    main(sys.argv[1:])
