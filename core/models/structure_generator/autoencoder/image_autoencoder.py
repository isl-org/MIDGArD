import logging
import torch

from core.models.neural_model import NeuralModel
from core.models.utils.vqvae.autoencoder import VQVAE
from core.utils.image_processor import ImageProcessor, ImgProcCfg


class ImageVQVAE(NeuralModel):
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
        neural_module = VQVAE(config["structure_generator"]["image_autoencoder"])
        super().__init__(config, neural_module)

        self.output_specs = {
            "metric": [
                "batch_loss",
                "loss_codebook",
                "loss_nll",
                "loss_image",
            ],
            "image": ["image", "gt_image"],
            "hist": [],
        }
        self.image_processor = ImageProcessor(ImgProcCfg())

    def generate_image(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Generates an image from the latent vector.

        Args:
            quant (torch.Tensor): Quantized latent tensor.

        Returns:
            torch.Tensor: The generated image.
        """
        neural_network = (
            self.neural_network.module
            if self.__dataparallel_flag__
            else self.neural_network
        )
        with torch.no_grad():
            image = neural_network.decode(quant)
        return image

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
                    data_batch["image"], data_batch["gt_image"] = [], []
                    for bid in range(n_batch):
                        image = self.generate_image(data_batch["z"][bid : bid + 1])
                        data_batch["image"].append(
                            self.image_processor.unnormalize(image[0])
                        )
                        data_batch["gt_image"].append(
                            self.image_processor.unnormalize(
                                data_batch["model_input"]["img"][bid]
                            )
                        )
                    if data_batch["image"]:
                        data_batch["image"] = torch.stack(data_batch["image"], 0)
                    if data_batch["gt_image"]:
                        data_batch["gt_image"] = torch.stack(data_batch["gt_image"], 0)

        except Exception as e:
            logging.error(f"Error in postprocessing after optimization: {e}")
            raise

        return data_batch
