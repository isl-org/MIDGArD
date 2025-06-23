"""
Taken from https://github.com/JiahuiLei/NAP/tree/main/logger
"""

from .base_logger import BaseLogger
import torch
import os
import open3d as o3d
import numpy as np


class MeshLogger(BaseLogger):
    def __init__(self, tb_logger, log_path, config) -> None:
        """
        Initialize the MeshLogger object.

        :param tb_logger: TensorBoard logger for logging mesh data.
        :param log_path: The logging directory path.
        :param config: Dictionary of configuration parameters.
        """
        super().__init__(tb_logger, log_path, config)
        self.NAME = "mesh"
        # Ensure the log directory exists
        os.makedirs(self.log_path, exist_ok=True)
        # Get configuration setting for visualizing one per batch
        self.viz_one = config["logging"]["viz_one_per_batch"]
        return

    def log_batch(self, batch) -> None:
        """
        Logs mesh data from the batch to TensorBoard and saves it to files.

        :param batch: Dictionary containing batch data, metadata, and configurations.
        """
        if self.NAME not in batch["output_parser"]:
            return

        keys_list = batch["output_parser"][self.NAME]
        if not keys_list:  # Exit if there are no meshes to process
            return

        if not batch["visualize"]:
            return  # Skip logging if visualization is not requested

        epoch_dir = os.path.join(self.log_path, f"epoch_{batch['epoch']}")
        os.makedirs(epoch_dir, exist_ok=True)

        data = batch["data"]
        phase = batch["phase"]
        meta_info = batch["meta_info"]

        for mesh_key in keys_list:  # for each key
            if mesh_key not in data:
                continue

            kdata = data[mesh_key]
            if isinstance(kdata, list):  # List of open3d mesh objects
                assert all(isinstance(m, o3d.geometry.TriangleMesh) for m in kdata)

                for i, mesh in enumerate(kdata):
                    viz_id = meta_info["viz_id"][i]

                    # Export .obj mesh files
                    mesh_file = f"{epoch_dir}/{mesh_key}_{viz_id}.obj"
                    o3d.io.write_triangle_mesh(mesh_file, mesh)

                    # Add mesh to TensorBoard
                    try:
                        self.tb.add_mesh(
                            tag=f"{mesh_key}/{phase}",
                            vertices=torch.tensor(mesh.vertices).unsqueeze(0).float(),
                            faces=torch.tensor(mesh.triangles).unsqueeze(0).int(),
                            global_step=batch["batch"],
                            config_dict=self._mesh_config(),  # Use a helper function for mesh config
                        )
                    except:
                        pass

                    if self.viz_one:
                        break  # Stop after one mesh if configured to do so

            elif isinstance(kdata, torch.Tensor):  # Tensor of point clouds
                assert kdata.dim() == 3 and kdata.size(2) in (
                    3,
                    6,
                ), "Point cloud logger accepts shape B,N,3/6"
                # Handle point clouds based on the provided features (3 or 6)
                for i, pc in enumerate(kdata):
                    self.tb.add_mesh(
                        tag=f"{mesh_key}/{phase}",
                        vertices=pc[..., :3].unsqueeze(0).float(),
                        colors=(
                            pc[..., 3:].unsqueeze(0).float()
                            if pc.size(2) == 6
                            else None
                        ),
                        global_step=batch["batch"],
                        config_dict={"material": {"cls": "PointsMaterial", "size": 10}},
                    )
                    viz_id = meta_info["viz_id"][i]
                    np.savetxt(
                        f"{epoch_dir}/{mesh_key}_{viz_id}.txt",
                        pc.detach().cpu().numpy(),
                    )

                    if self.viz_one:
                        break  # Stop after one mesh if configured to do so

            else:
                raise ValueError(f"Unsupported data type for mesh_key '{mesh_key}'")

    def log_phase(self) -> None:
        """
        Placeholder for logging at the end of each phase.
        """
        pass

    def _mesh_config(self):
        """
        Helper function that returns the default configuration dictionary for the mesh.

        :return: Default mesh configuration settings.
        """
        return {
            "camera": {"cls": "PerspectiveCamera", "fov": 75},
            "lights": [
                {"cls": "AmbientLight", "color": "#ffffff", "intensity": 0.7},
                {
                    "cls": "DirectionalLight",
                    "color": "#ffffff",
                    "intensity": 0.65,
                    "position": [0, 2, 0],
                },
            ],
            "material": {"cls": "MeshStandardMaterial", "roughness": 1, "metalness": 0},
        }
