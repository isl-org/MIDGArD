from copy import deepcopy
from functools import partial
import logging
import numpy as np
import open3d as o3d
import os
import platform
import torch

from core.models.neural_model import NeuralModel
from core.models.utils.common import MLP
from core.models.shape_generator.denoiser.openai_model_3d import DiffusionUNet
from core.models.shape_generator.autoencoder.shape_autoencoder import ShapeVQVAE
from core.models.shape_generator.condition.bert.network import BERTTextEncoder
from core.models.shape_generator.condition.resnet_v1 import resnet18
from core.models.shape_generator.condition.gat import GAT
from core.utils.mesh_extractor import MeshExtractor
from core.models.utils.ldm_diffusion_util import (
    make_beta_schedule,
    extract_into_tensor,
    exists,
    default,
)
from core.models.utils.samplers.ddim import DDIMSampler
from core.models.utils.distributed import reduce_loss_dict


class ShapeGenerator(NeuralModel):
    """
    Specialization of the NeuralModel class to ShapeGenerator
    """

    def __init__(self, config):
        """
        Initializes the ShapeGenerator with configuration settings.

        Args:
            config (Dict): Configuration dictionary for the model setup.
        """
        neural_module = SDFusionDDM(config)
        super().__init__(config, neural_module)

        # Configuration variables
        self.viz_one = config["logging"]["viz_one_per_batch"]
        self.iou_threshold = config["evaluation"]["iou_threshold"]
        self.viz_dpi = config["logging"].get("viz_dpi", 200)
        self.mesh_extractor = MeshExtractor()

        # Define output specs
        self.output_specs = {
            "metric": [
                "batch_loss",
                "loss_simple",
                "loss_gamma",
                "logvar",
                "loss_vlb",
                "loss_total",
            ],
            "mesh": ["mesh", "mesh_gt"],  # , "mesh_ae"
        }

    def generate_mesh_from_sdf(self, sdf: torch.Tensor) -> o3d.geometry.TriangleMesh:
        """
        Generates a 3D mesh from a Signed Distance Field.

        Args:
            sdf (torch.Tensor): Signed distance field.

        Returns:
            o3d.geometry.TriangleMesh: The generated 3D mesh.
        """

        mesh = self.mesh_extractor.generate_from_sdf(sdf, level=self.config["dataset"].get("level", 0.02))

        if mesh == None:
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=1.0, height=1.0, depth=1.0
            )
            logging.warning("Mesh extraction fail, replace by a place holder")
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
                is_training = self.neural_network.training

                # Set the network in evaluation mode to generate visualization
                self.neural_network.eval()

                n_batch = data_batch["gen_sdf"].shape[0]
                with torch.no_grad():
                    data_batch["mesh"] = []
                    data_batch["mesh_gt"] = []
                    # data_batch["mesh_ae"] = []
                    for bid in range(n_batch):
                        mesh = self.generate_mesh_from_sdf(
                            sdf=data_batch["gen_sdf"][bid : bid + 1]
                        )
                        mesh_gt = self.generate_mesh_from_sdf(
                            sdf=data_batch["gt_sdf"][bid : bid + 1]
                        )
                        # mesh_ae = self.generate_mesh_from_sdf(
                        #     sdf=data_batch["ae_sdf"][bid : bid + 1]
                        # )
                        data_batch["mesh"].append(mesh)
                        data_batch["mesh_gt"].append(mesh_gt)
                        # data_batch["mesh_ae"].append(mesh_ae)

                        if self.viz_one:
                            break

                # Move back to training mode if initially training
                if is_training:
                    self.neural_network.train()

        except Exception as e:
            logging.error(f"Error in postprocessing after optimization: {e}")
            raise

        return data_batch


class SDFusionDDM(torch.nn.Module):
    """
    An implementation of the SDFusion Denoising Diffusion Model (DDM)
    """

    def __init__(self, config) -> None:
        """
        Initializes the SDFusionDDM network with the given configuration.

        Args:
            config (Dict): Configuration dictionary for the network.
        """
        super().__init__()

        self.network_dict = torch.nn.ModuleDict()
        self.config = deepcopy(config)
        self.condition_type = self.config["shape_generator"]["denoising_diffusion"].get(
            "condition_type", []
        )
        self.subtract_bb = self.config["shape_generator"]["denoising_diffusion"].get(
            "subtract_bb", True
        )
        self.cond_bb = (
            True
            if self.subtract_bb
            else self.config["shape_generator"]["denoising_diffusion"].get(
                "cond_bb", True
            )
        )
        print("SUBTRACT BB & COND BB", self.subtract_bb, self.cond_bb)
        self.bb_prior_factor = self.config["shape_generator"][
            "denoising_diffusion"
        ].get("bb_prior_factor", 1)

        if self.cond_bb:
            self.config["shape_generator"]["unet_3d"]["in_channels"] = 6

        # Initialize the VQVAE
        self._init_shape_autoencoder()

        # Initialize the diffusion process
        self._init_diffusion()

        # Initialize the sampler
        self.ddim_sampler = DDIMSampler(self, schedule="linear", device=self.device)

        if "img" in self.condition_type:
            # Initialize the Image Condition model
            self.network_dict["img_enc"] = self._init_image_condition()
            self.network_dict["img_linear"] = self._init_image_linear()

        if "txt" in self.condition_type:
            # Initialize the Text Condition model
            self.network_dict["txt_enc"] = self._init_text_condition()

        if "graph" in self.condition_type:
            # Initialize the Graph Condition model
            self.network_dict["graph_enc"] = self._init_graph_condition()

        # Initialize the UNet denoising network used in the diffusion process
        self.network_dict["denoiser"] = self._init_denoiser()

    def _init_diffusion(self) -> None:
        """
        Initialize the Diffusion model.
        """
        os_type = platform.system()
        if os_type == "Darwin":
            self.device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        elif os_type == "Linux" or os_type == "Windows":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.xpu.is_available():
                self.device = torch.device("xpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        self.parameterization = "eps"
        self.learn_logvar = False
        self.v_posterior = 0.0
        self.original_elbo_weight = 0.0
        self.l_simple_weight = 1.0
        self.ddim_steps = 100

        timesteps = self.config["shape_generator"]["denoising_diffusion"].get(
            "time_steps", 1000
        )
        beta_schedule = self.config["shape_generator"]["denoising_diffusion"].get(
            "schedule_type", "linear"
        )
        linear_start = self.config["shape_generator"]["denoising_diffusion"].get(
            "beta_start", 0.00085
        )
        linear_end = self.config["shape_generator"]["denoising_diffusion"].get(
            "beta_end", 0.012
        )
        self.register_schedule(
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        logvar_init = 0.0
        self.logvar = torch.full(
            fill_value=logvar_init, size=(self.num_timesteps,), device=self.device
        )
        self.uc_scale = 1.0

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ) -> None:
        """
        Registers the noise schedule for the diffusion process based on given or calculated betas.

        This function initializes and stores various tensors required for the diffusion process,
        including the noise schedule (betas), the cumulative product of alphas (1 - betas), and
        several other tensors derived from these for use in diffusion calculations and posterior
        sampling.

        Args:
            given_betas (np.ndarray, optional): An array of beta values to use directly. If not provided,
                betas are generated based on the specified schedule.
            beta_schedule (str): The type of schedule to generate beta values, if `given_betas` is None.
                Defaults to "linear".
            timesteps (int): The number of timesteps in the diffusion process. Defaults to 1000.
            linear_start (float): The start value for linearly spaced betas. Used if `beta_schedule` is "linear".
            linear_end (float): The end value for linearly spaced betas. Used if `beta_schedule` is "linear".
            cosine_s (float): The s parameter for the cosine schedule. Used if `beta_schedule` is "cosine".

        The method supports linear and cosine schedules for beta generation and performs assertions to
        ensure the correct shape and size of the generated arrays. It converts all necessary components
        into PyTorch tensors and moves them to the specified device for GPU acceleration.
        """
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(
                self.betas**2
                / (
                    2
                    * self.posterior_variance
                    * to_torch(alphas)
                    * (1 - self.alphas_cumprod)
                )
            )
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def _init_shape_autoencoder(self) -> None:
        """
        Initialize the Signed Distance Field (SDF) decoder
        """
        sdf_decoder_config = self.config["shape_generator"]["shape_autoencoder"]

        # Shape decoder network
        self.grid_resolution = sdf_decoder_config["ddconfig"].get("resolution", 64)
        z_ch = sdf_decoder_config["ddconfig"].get("z_channels", 3)
        n_down = len(sdf_decoder_config["ddconfig"].get("ch_mult", [1, 2, 4])) - 1
        z_sp_dim = self.grid_resolution // (2**n_down)
        self.z_shape = (z_ch, z_sp_dim, z_sp_dim, z_sp_dim)
        self.vqvae = ShapeVQVAE(self.config)
        ckpt_fn = os.path.join(
            self.config["base_directory"],
            self.config["output_directory"],
            sdf_decoder_config.get("checkpoint_path", ""),
        )
        ckpt = torch.load(ckpt_fn, map_location="cpu")
        self.vqvae.model_resume(ckpt, is_initialization=True, network_name=["all"])
        logging.info("Loaded sdf decoder weights from %s", ckpt_fn)

        # Copy model to GPU memory
        self.vqvae.to_gpus()

        # Set in evaluation mode to save memory
        self.vqvae.set_eval()

    def _init_denoiser(self) -> torch.nn.Module:
        """
        Initialize the denoising network
        """
        # 3D UNet denoiser:
        denoiser_config = self.config["shape_generator"]["unet_3d"]
        cond_key = self.config["shape_generator"]["denoising_diffusion"].get(
            "conditioning_key", "crossattn"
        )
        return DiffusionUNet(denoiser_config, conditioning_key=cond_key)

    def _init_image_linear(self) -> torch.nn.Module:
        """
        Initialize the Text Condition model.
        """
        img_context_d = 512
        txt_context_d = self.config["shape_generator"]["bert"].get("n_embed", 1280)
        img_lin = torch.nn.Linear(img_context_d, txt_context_d)
        for param in img_lin.parameters():
            param.requires_grad = True
        return img_lin

    def _init_image_condition(self) -> torch.nn.Module:
        """
        Initialize the Text Condition model.
        """
        img_enc = resnet18(pretrained=True)  # context dim: 512
        for param in img_enc.parameters():
            param.requires_grad = True
        return img_enc

    def _init_text_condition(self) -> torch.nn.Module:
        """
        Initialize the Text Condition model.
        """
        bert_params = self.config["shape_generator"]["bert"]
        self.text_embed_dim = bert_params.get("n_embed", 1280)
        txt_enc = BERTTextEncoder(
            n_embed=bert_params.get("n_embed", 1280),
            n_layer=bert_params.get("n_layer", 32),
            device=self.device,
        )
        for param in txt_enc.parameters():
            param.requires_grad = True
        return txt_enc

    def _init_graph_condition(self) -> torch.nn.Module:
        """
        Initialize the Grpah Condition model.
        """
        model_params = self.config["shape_generator"]["graphmodel"]
        if model_params["architecture"] == "mlp":
            bb_enc = MLP(
                in_dim=model_params.get("in_dim", 3),
                out_dim=model_params.get("out_dim", 1280),
                hidden_dims=model_params.get("hidden_dims", [16, 64, 256]),
                use_batch_normalization=model_params.get(
                    "use_batch_normalization", True
                ),
            )
        elif model_params["architecture"] == "gat_local":
            bb_enc = GAT(
                in_channels=model_params.get("in_dim", 3),
                hidden_channels=model_params.get("gat_hidden_channels", 8),
                out_channels=model_params.get("gat_out_channels", 8),
                heads=model_params.get("gat_heads", 8),
                num_classes=model_params.get("out_dim", 1280),
            )
        elif model_params["architecture"] == "gat_global":
            bb_enc = GAT(
                in_channels=model_params.get("in_dim", 4),
                hidden_channels=model_params.get("gat_hidden_channels", 8),
                out_channels=model_params.get("gat_out_channels", 8),
                heads=model_params.get("gat_heads", 8),
                num_classes=model_params.get("out_dim", 1280),
            )
        else:
            raise NotImplementedError("only mlp and gat implemented")
        for param in bb_enc.parameters():
            param.requires_grad = True
        return bb_enc

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        """
        Applies the denoising model to the input noisy data with optional conditioning.

        This function wraps the call to the network's denoiser, which is a U-Net model configured
        for the diffusion process. It handles different types of conditioning based on the model configuration.

        Args:
            x_noisy (torch.Tensor): The noisy input data tensor.
            t (torch.Tensor): The current timestep tensor, indicating the diffusion step.
            cond (dict, list, or torch.Tensor): The conditioning data. The format and usage depend on the
                conditioning mechanism specified by the model's configuration.
            return_ids (bool): If True and the model output is a tuple, only the first element of the tuple
                is returned. Defaults to False.

        Returns:
            torch.Tensor or tuple: The output from the denoiser. If `return_ids` is False and the output
            is a tuple, only the first element of the tuple is returned.

        The method supports multiple conditioning mechanisms: concatenation, cross-attention, hybrid (both
        concatenation and cross-attention), and adm (a specific type of conditioning used in ADM models).
        """
        # Handle conditioning data format
        if isinstance(cond, dict):
            pass  # No action needed if cond is already a dictionary
        else:
            # Convert non-dict conditioning into a dictionary format
            if not isinstance(cond, list):
                cond = [cond]
            key = (
                "c_concat"
                if self.network_dict["denoiser"].conditioning_key == "concat"
                else "c_crossattn"
            )
            cond = {key: cond}

        # Apply the denoiser model based on conditioning configuration
        out = self.network_dict["denoiser"](x_noisy, t, **cond)

        # Handle model output, returning only the necessary part if specified
        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out

    def get_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str = "l2",
        mean: bool = True,
    ):
        """
        Computes the loss between predictions and targets using specified loss type.

        Args:
            pred (torch.Tensor): The predicted values.
            target (torch.Tensor): The ground truth values.
            loss_type (str): The type of loss function to use. Supported types are "l1" and "l2".
            mean (bool): If True, returns the mean loss; otherwise, returns the loss per element.

        Returns:
            torch.Tensor: The computed loss.

        This method supports L1 (absolute difference) and L2 (mean squared error) loss calculations,
        providing flexibility for model training and evaluation.
        """
        if loss_type == "l1":
            # Compute L1 loss
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == "l2":
            # Compute L2 loss with optional reduction
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def q_sample(self, x_start, t, noise=None):
        """
        Samples noisy versions of the input data at a specific timestep during the diffusion process.

        Args:
            x_start (torch.Tensor): The original, clean data tensor.
            t (torch.Tensor): The timestep tensor indicating the current step in the diffusion process.
            noise (torch.Tensor, optional): An externally provided noise tensor. If not provided, noise
                is generated internally.

        Returns:
            torch.Tensor: The noisy data tensor after applying the diffusion noise model.

        This method applies the diffusion noise model to the input data, generating a noisy version
        based on the specified timestep. It supports externally provided noise for flexibility in noise
        application scenarios.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_v(self, x, noise, t):
        """
        Get the v term for the diffusion process.

        Args:
            x (torch.Tensor): The input tensor.
            noise (torch.Tensor): The noise tensor.
            t (torch.Tensor): The timestep tensor.

        Returns:
            torch.Tensor: The computed v term.
        """
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def p_losses(
        self,
        z_start: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
        z_cond: torch.Tensor = None,
    ):
        """
        Computes prediction losses for the diffusion process.

        This method calculates the loss based on the prediction error of the model's output
        compared to the target, which can be either the original input (`x0`) or the added noise (`eps`),
        depending on the model's parameterization.

        Args:
            z_start (torch.Tensor): The original input tensor.
            cond (torch.Tensor): The conditioning tensor.
            t (torch.Tensor): The timestep tensor indicating the current step in the diffusion process.
            noise (torch.Tensor, optional): The noise tensor. If not provided, it is generated.
            z_cond (torch.Tensor, optional): The conditioned input tensor. If not provided, it is generated.

        Returns:
            tuple: A tuple containing:
                - x_noisy (torch.Tensor): The noisy input tensor after adding diffusion noise.
                - target (torch.Tensor): The target tensor for loss computation.
                - loss (torch.Tensor): The computed total loss.
                - loss_dict (dict): A dictionary of individual loss components.

        The method supports two parameterizations: predicting the original input (`x0`)
        and predicting the noise (`eps`). It calculates a simple L2 loss and a variational lower bound (VLB) loss,
        optionally adjusting for learned log variance.
        """
        # Default noise to random if not specified
        noise = default(noise, lambda: torch.randn_like(z_start))

        # Apply diffusion noise to input
        z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)

        # Concat with conditioning
        if z_cond is not None:
            model_inp = torch.cat([z_noisy, z_cond], dim=1)
        else:
            model_inp = z_noisy

        # Model predicts either noise (eps) or original input (x0)
        model_output = self.apply_model(model_inp, t, cond)

        # Initialize dictionary to store loss components
        loss_dict = {}

        # Determine the target based on the model's parameterization
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = z_start
        elif self.parameterization == "v":
            target = self.get_v(z_start, noise, t)
        else:
            raise NotImplementedError(
                f"Parameterization {self.parameterization} not yet supported"
            )

        # Compute simple L2 loss
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f"loss_simple": loss_simple.mean()})

        # Adjust loss with log variance if applicable
        logvar_t = self.logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({"loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        # Scale simple loss
        loss = self.l_simple_weight * loss.mean()

        # Compute variational lower bound (VLB) loss
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(
            dim=(1, 2, 3, 4)
        )
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({"loss_vlb": loss_vlb})

        # Add VLB to total loss
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({"loss_total": loss.clone().detach().mean()})

        return z_noisy, target, loss, loss_dict

    def process_condition(
        self,
        key,
        data_batch,
        scale,
        condition_func,
        post_func=None,
        inference=False,
    ):
        """
        Processes the condition data based on the specified key and model configuration.

        Args:
            key (str): The condition key indicating the type of condition data.
            data_batch (Dict): The input data batch containing the condition data.
            scale (Dict): The scaling factors for the condition data.
            condition_func (torch.nn.Module): The condition model function.
            post_func (torch.nn.Module, optional): The post-processing function for the condition data.
            inference (bool): Flag indicating whether the process is for inference.

        Returns:
            tuple: A tuple containing:
                - c (torch.Tensor): The processed condition tensor.
                - uc (torch.Tensor): The processed unconditional condition tensor.
        """
        data = data_batch[key]
        B = data_batch["sdf"].shape[0]

        bb_is_graph = (
            "gat" in self.config["shape_generator"]["graphmodel"]["architecture"]
        )
        if key == "img":
            null_in = torch.zeros_like(data).to(self.device)
        elif key == "graph" and not bb_is_graph:
            null_in = torch.zeros_like(data).to(self.device, dtype=torch.float32)
        elif key == "graph" and bb_is_graph:
            null_in = data.detach().clone()
            null_in.x = torch.zeros_like(data.x)
            null_in = null_in.to(self.device, dtype=torch.float32)
        else:
            null_in = B * [""]

        # Apply input layers to the condition signal
        uc = condition_func(null_in)

        # Apply input layers to the condition signal
        c = condition_func(data)

        # Post processing for images
        if post_func:
            c = post_func(c)
            uc = post_func(uc)

        if c.dim() == 2:  # for bb, we need to unsqeeze
            c = c.unsqueeze(1)

        if uc.dim() == 2:  # for bb, we need to unsqeeze
            uc = uc.unsqueeze(1)

        if not inference:  # Dropout only during training
            p = torch.rand(B, device=self.device) > 0.5  # Random mask
            c = c * p[:, None, None]
        else:
            c = scale[key] * c
            uc = scale[key] * uc

        return c, uc

    def process_conditions(self, data_batch, scale, inference=False):
        """
        Processes the condition data based on the model configuration.

        Args:
            data_batch (Dict): The input data batch containing the condition data.
            scale (Dict): The scaling factors for the condition data.
            inference (bool): Flag indicating whether the process is for inference.

        Returns:
            tuple: A tuple containing:
                - c_mm (torch.Tensor): The processed condition tensor.
                - uc_mm (torch.Tensor): The processed unconditional condition tensor.
        """
        c_mm_parts = []
        uc_mm_parts = []

        for key in ("txt", "img", "graph"):
            if key in self.condition_type:
                condition_func = self.network_dict[f"{key}_enc"]
                post_func = self.network_dict["img_linear"] if key == "img" else None
                c, uc = self.process_condition(
                    key, data_batch, scale, condition_func, post_func, inference
                )

                c_mm_parts.append(c)
                uc_mm_parts.append(uc)

        return torch.cat(c_mm_parts, dim=1), torch.cat(uc_mm_parts, dim=1)

    def forward(self, data_batch, visualize):
        """
        Executes the forward pass of the network, including encoding, diffusion, and loss computation.

        Args:
            data_batch (Dict): The input data batch containing the signed distance field (SDF) values.
            visualize (bool): Flag indicating whether to generate visualization data.

        Returns:
            Dict: The output data pack containing predictions, loss values, and potentially visualization data.

        This method orchestrates the forward pass, involving the encoding of input SDF values to a latent
        representation, performing a diffusion process to introduce noise, computing losses, and preparing
        output data, including loss metrics and visualization information if requested.
        """
        output = {}
        output["visualize"] = visualize

        # Extract SDF values from the input data batch
        sdf = data_batch["sdf"]
        B = sdf.shape[0]
        scale = {}
        scale["uc"] = self.uc_scale
        for key in self.condition_type:
            scale[key] = 1.0

        # as float: bb inside 0, outside 1 (because of how it is currently represented)
        bb_as_sdf = data_batch["bounding_box_sdf"]
        if data_batch["phase"] == "train":
            # Set denoiser to training mode
            for key in self.network_dict.keys():
                try:
                    self.network_dict[key].set_train()
                except:
                    self.network_dict[key].train()

            # Condition
            c_mm, _ = self.process_conditions(data_batch, scale, inference=False)

            # Encode bounding box to latent space
            with torch.no_grad():
                z = self.vqvae.neural_network.encode_no_quant(sdf).detach()
                z_cond = None
                if self.cond_bb:
                    z_cond = self.vqvae.neural_network.encode_no_quant(
                        bb_as_sdf
                    ).detach()
                    if self.subtract_bb:
                        z = z - z_cond

            # Sample a random timestep for the diffusion process
            t = torch.randint(
                0, self.num_timesteps, (z.shape[0],), device=self.device
            ).long()

            # Compute losses for the diffusion process
            z_noisy, z_target, loss, loss_dict = self.p_losses(
                z, c_mm, t, z_cond=z_cond
            )

            # Aggregate computed losses for backpropagation
            loss_dict = reduce_loss_dict(loss_dict)
            output.update(loss_dict)
            output["batch_loss"] = loss

        if (
            (data_batch["phase"] == "val")
            or (data_batch["phase"] == "test")
            or visualize
        ):
            # Condition
            c_mm, mm_uc_feat = self.process_conditions(
                data_batch, scale, inference=True
            )

            # Set denoiser to eval mode
            for key in self.network_dict.keys():
                try:
                    self.network_dict[key].set_eval()
                except:
                    self.network_dict[key].eval()

            # Perform DDIM sampling
            z_cond = None
            if self.cond_bb:
                z_cond = self.vqvae.neural_network.encode_no_quant(bb_as_sdf).detach()

            samples, intermediates = self.ddim_sampler.sample(
                S=self.ddim_steps,
                batch_size=B,
                shape=self.z_shape,
                conditioning=c_mm,
                verbose=False,
                unconditional_guidance_scale=scale["uc"],
                unconditional_conditioning=mm_uc_feat,
                part_conditioning=z_cond,
                eta=0.0,
                quantize_x0=False,
            )

            if self.subtract_bb:
                samples = samples + z_cond

            # Decode the sampled tensor to get the completed shape
            if visualize:
                output["gen_sdf"] = self.vqvae.neural_network.decode_no_quant(samples)
                # quant, _, _ = self.vqvae.neural_network.encode(sdf)
                # output["ae_sdf"] = self.vqvae.neural_network.decode(quant)
                output["gt_sdf"] = sdf

            # Set denoiser back to training mode
            for key in self.network_dict.keys():
                try:
                    self.network_dict[key].set_train()
                except:
                    self.network_dict[key].train()

        return output

    @torch.no_grad()
    def mm_inference(
        self, data_batch, scale, ddim_steps=None, ddim_eta=0.0, uc_scale=None
    ):
        """
        Performs multimodal inference using the trained model.

        Args:
            data_batch (Dict): The input data containing the signed distance field (SDF) values.
            scale (Dict): The scaling factors for the condition data.
            ddim_steps (int): The number of steps to perform during DDIM sampling. Defaults to None.
            ddim_eta (float): The eta value for DDIM sampling. Defaults to 0.0.
            uc_scale (float): The scaling factor for unconditional condition data. Defaults to None.

        Returns:
            torch.Tensor: The generated SDF tensor.
        """
        for key in self.network_dict.keys():
            try:
                self.network_dict[key].set_eval()
            except:
                self.network_dict[key].eval()

        B = data_batch["sdf"].shape[0]

        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.uc_scale

        # Condition
        c_mm, mm_uc_feat = self.process_conditions(data_batch, scale, inference=True)

        # Perform DDIM sampling
        z_cond = []
        if self.cond_bb:
            z_cond = self.vqvae.neural_network.encode_no_quant(
                data_batch["bb_3D"].unsqueeze(1)
            ).detach()

        samples, _ = self.ddim_sampler.sample(
            S=ddim_steps,
            batch_size=B,
            shape=self.z_shape,
            conditioning=c_mm,
            verbose=False,
            unconditional_guidance_scale=uc_scale,
            unconditional_conditioning=mm_uc_feat,
            part_conditioning=z_cond,
            eta=ddim_eta,
            quantize_x0=False,
        )
        if self.subtract_bb:
            samples = samples + z_cond

        return self.vqvae.neural_network.decode_no_quant(samples)
