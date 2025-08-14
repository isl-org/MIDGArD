import logging
import open3d as o3d
import torch

from core.models.neural_model import NeuralModel
from core.models.utils.vqvae.autoencoder import VQVAE
from core.utils.mesh_extractor import MeshExtractor


class ShapeVQVAE(NeuralModel):
    """
    Wrapper for the unified VQVAE network with additional postprocessing.
    """

    def __init__(self, config: dict) -> None:
        """
        Instantiates the unified VQVAE module.

        Args:
            config (dict): The experiment configuration.

        Returns:
            None
        """
        neural_module = VQVAE(config["shape_generator"]["shape_autoencoder"])
        super().__init__(config, neural_module)

        self.output_specs = {
            "metric": [
                "batch_loss",
                "loss_codebook",
                "loss_nll",
                "loss_sdf",
            ],
            "mesh": ["mesh", "gt_mesh"],
        }
        self.mesh_extractor = MeshExtractor()

    def generate_mesh_from_latent(
        self, quant: torch.Tensor
    ) -> o3d.geometry.TriangleMesh:
        """
        Generates a 3D mesh from the latent vector.

        Args:
            quant (torch.Tensor): Quantized latent tensor.

        Returns:
            o3d.geometry.TriangleMesh: The generated 3D mesh.
        """
        neural_network = (
            self.neural_network.module
            if self.__dataparallel_flag__
            else self.neural_network
        )

        sdf = neural_network.decode(quant)
        mesh = self.mesh_extractor.generate_from_sdf(sdf)
        if mesh is None:
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=1.0, height=1.0, depth=1.0
            )
            logging.warning("Mesh extraction failed, using placeholder.")
        return mesh

    def generate_mesh_from_sdf(self, sdf: torch.Tensor) -> o3d.geometry.TriangleMesh:
        """
        Generates a 3D mesh from a Signed Distance Field.

        Args:
            sdf (torch.Tensor): Signed distance field.

        Returns:
            o3d.geometry.TriangleMesh: The generated 3D mesh.
        """

        mesh = self.mesh_extractor.generate_from_sdf(sdf)
        if mesh is None:
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=1.0, height=1.0, depth=1.0
            )
            logging.warning("Mesh extraction failed, using placeholder.")
        return mesh

    def _postprocess_after_optim(self, data_batch: dict) -> dict:
        """
        Postprocesses outputs after an optimization step for visualization.

        Args:
            data_batch (dict): Dictionary with output data from the forward pass.

        Returns:
            dict: Updated data_batch including visualization outputs (meshes or images).
        """
        try:
            if data_batch["visualize"]:
                n_batch = data_batch["z"].shape[0]
                with torch.no_grad():
                    data_batch["mesh"] = []
                    data_batch["gt_mesh"] = []
                    for bid in range(n_batch):
                        mesh = self.generate_mesh_from_latent(
                            data_batch["z"][bid : bid + 1]
                        )
                        gt_mesh = self.generate_mesh_from_sdf(
                            data_batch["model_input"]["sdf"][bid : bid + 1]
                        )
                        data_batch["mesh"].append(mesh)
                        data_batch["gt_mesh"].append(gt_mesh)

        except Exception as e:
            logging.error(f"Error in postprocessing after optimization: {e}")
            raise

        return data_batch
