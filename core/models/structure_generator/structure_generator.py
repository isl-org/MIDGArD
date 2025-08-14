from copy import deepcopy
import logging
import networkx as nx
import numpy as np
import os
from scipy.sparse.csgraph import minimum_spanning_tree
import torch
import torchvision
import trimesh
from tqdm import tqdm
from core.models.structure_generator.denoiser.graph_transformer import (
    GraphTransformer,
)
from core.models.neural_model import NeuralModel
from core.models.structure_generator.autoencoder.image_autoencoder import ImageVQVAE
from core.utils.data_utils import get_G_from_VE
from core.utils.visualization_utils import viz_G_topology, viz_G_BB


class StructureGenerator(NeuralModel):
    """
    Specialization of the NeuralModel class to StructureGenerator
    """

    def __init__(self, config) -> None:
        """
        Initializes the StructureGenerator with configuration settings.

        Args:
            config (Dict): Configuration dictionary for the model setup.
        """
        neural_module = NeuralArticulationDDM(config)
        super().__init__(config, neural_module)

        # Configuration variables
        self.viz_one = config["logging"].get("viz_one_per_batch")
        self.iou_threshold = config["evaluation"].get("iou_threshold")
        self.viz_dpi = config["logging"].get("viz_dpi", 200)
        self.viz_frame_N = config["logging"].get("viz_frame", 10)
        self.N_gen = 2
        self.max_nodes = config["dataset"].get("max_nodes", 8)
        self.body_categories = config.get("body_categories", None)
        self.asset_categories = config.get("asset_categories", None)

        # Ablations
        self.ablations = neural_module.ablations
        self.feature_size = neural_module.feature_size
        self.feature_index = neural_module.feature_index

        # Image decoder
        image_decoder_config = config["structure_generator"]["image_autoencoder"]
        self.input_image_hw = image_decoder_config["ddconfig"].get("resolution", 256)
        self.output_channels = image_decoder_config["ddconfig"].get("out_ch", 3)  # rgb
        self.output_image_hw = image_decoder_config.get("output_hw", 137)

        # Sanity checks
        assert (
            len(self.asset_categories) == self.feature_size["node_sa"]
        ), f"{len(self.asset_categories)} vs {self.feature_size['node_sa']}"
        assert (
            len(self.body_categories) == self.feature_size["node_sb"]
        ), f"{len(self.body_categories)} vs {self.feature_size['node_sb']}"

        # Define output specs based on ablations
        self.output_specs = {
            "metric": [
                "batch_loss",
                "loss_e",
                "loss_v",
                "loss_v_oc",
                "loss_v_bb",
                "loss_v_t",
                "loss_e_p",
                "loss_e_l",
            ],
            "image": ["viz_gt", "viz_gen"],
            "video": ["viz_gt_vid", "viz_gen_vid"],
            "mesh": ["input"],
            "hist": ["loss_v_i", "loss_e_i"],
            "xls": [],
        }

        # Add conditional metrics based on ablations
        if self.ablations["use_categorical_asset"]:
            self.output_specs["metric"].append("loss_v_sa")
        if self.ablations["use_categorical_body"]:
            self.output_specs["metric"].append("loss_v_sb")
        if self.ablations["use_categorical_joint"]:
            self.output_specs["metric"].append("loss_e_j")
        if self.ablations["use_node_orientation"]:
            self.output_specs["metric"].append("loss_v_r")
        if self.ablations["use_manifold_plucker"]:
            self.output_specs["metric"].append("loss_e_m")
        if self.ablations["use_node_2D_preencoded_latent"]:
            self.output_specs["metric"].append("loss_v_2d")

    def _extract_mesh_for_G(self, G, gt_mesh_list=None):
        """
        Extracts and processes mesh data for each node in the graph G.

        This function iterates through all nodes in the graph G. For each node, it extracts the 'additional' attribute,
        generates a mesh using this attribute, and then applies various transformations to this mesh, such as translation
        and scaling, to normalize its position and size. If a list of ground truth meshes is provided, it processes these
        meshes in a similar manner and associates them with the corresponding nodes in G.

        Args:
            G: The graph containing nodes with mesh information.
            gt_mesh_list (List, optional): A list of ground truth meshes corresponding to each node in G.

        Returns:
            The modified graph G with updated mesh attributes for each node.
        """
        logging.info("Extract mesh for G ...")
        for node, node_data in G.nodes(data=True):
            if "additional" not in node_data:
                continue

            bbox = node_data["bbox"].copy()
            mesh = trimesh.primitives.Box(extents=2 * abs(bbox), mutable=True)
            mesh_centroid = mesh.bounds.mean(0)
            mesh.apply_translation(-mesh_centroid)
            scale = (
                2.0
                * np.linalg.norm(bbox)
                / np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
            )
            mesh.apply_scale(scale)
            node_data["mesh"] = mesh
            nx.set_node_attributes(G, {node: {"mesh": mesh}})

            if gt_mesh_list is not None:
                gt_mesh = gt_mesh_list[node].copy()
                mesh_centroid = gt_mesh.bounds.mean(0)
                gt_mesh.apply_translation(-mesh_centroid)
                scale = (
                    2.0
                    * np.linalg.norm(bbox)
                    / np.linalg.norm(gt_mesh.bounds[1] - gt_mesh.bounds[0])
                )
                gt_mesh.apply_scale(scale)
                nx.set_node_attributes(G, {node: {"gt_mesh": gt_mesh}})

        return G

    def _generate_object(self, node_scale, edge_scale):
        """
        Run the diffusion model to generate an object graph.

        Args:
            node_scale: Scaling factor for vertices.
            edge_scale: Scaling factor for edges.

        Returns:
            Tuple: A tuple containing two elements; the first is a tensor representing the nodes of the generated object, and the second is a tensor representing the edges.
        """
        neural_network = (
            self.neural_network.module
            if self.__dataparallel_flag__
            else self.neural_network
        )

        # Start from random node and edge feature vectors
        node_features = torch.randn(
            self.N_gen, self.max_nodes, self.feature_size["node"]
        ).to(self.device)
        edge_features = torch.randn(
            self.N_gen,
            (self.max_nodes * (self.max_nodes - 1)) // 2,
            self.feature_size["edge"],
        ).to(self.device)

        # For a masked generation process
        if neural_network.use_hard_node_mask:
            random_n = torch.randint(low=2, high=self.max_nodes + 1, size=(self.N_gen,))
            node_mask = torch.arange(self.max_nodes)[None, :] < random_n[:, None]
            node_mask = node_mask.float().to(node_features.device)
            # Set the noisy first channel to ground truth node_mask
            node_features[..., 0] = node_mask
        else:
            node_mask = None

        # Run the diffusion model
        nodes, edges = neural_network.generate(
            node_features,
            edge_features,
            node_scale,
            edge_scale,
            node_mask=node_mask,
        )

        return nodes, edges

    def _postprocess_after_optim(self, data_batch: dict) -> dict:
        """
        Additional postprocessing after optimizer.step

        Args:
            data_batch (dict): The postprocessed prediction.

        Returns:
            dict: Further processed prediction.
        """
        try:
            if data_batch["visualize"]:
                is_training = self.neural_network.training

                # Set the network in evaluation mode to generate visualization
                self.neural_network.eval()

                viz_pred_list = []
                viz_gt_list = []
                viz_pred_vid_list = []
                viz_gt_vid_list = []

                # Only visualize one ground truth
                rgb_gt, rgb_gt_list = self._visualize_graph(
                    data_batch["V_gt"][0], data_batch["E_gt"][0], title="gt"
                )
                viz_gt_list.append(rgb_gt_list[0].transpose(2, 0, 1))
                viz_gt_vid_list.append(np.stack(rgb_gt_list, 0).transpose(0, 3, 1, 2))

                # Visualization generation
                gen_V, gen_E = self._generate_object(
                    data_batch["V_scale"], data_batch["E_scale"]
                )
                batch_size = len(gen_V)

                if self.viz_one:
                    _iter = np.random.permutation(batch_size)
                else:
                    _iter = range(batch_size)
                for batch_id in _iter:
                    batch_id = int(batch_id)
                    rgb_pred, rgb_pred_list = self._visualize_graph(
                        gen_V[batch_id], gen_E[batch_id], title="pred"
                    )
                    viz_pred_list.append(rgb_pred_list[0].transpose(2, 0, 1))
                    viz_pred_vid_list.append(
                        np.stack(rgb_pred_list, 0).transpose(0, 3, 1, 2)
                    )
                    if self.viz_one:
                        break

                if len(viz_gt_list) > 0:
                    data_batch["viz_gt"] = torch.Tensor(
                        np.stack(viz_gt_list, axis=0)
                    )  # batch_size, 3, height, width
                    data_batch["viz_gt_vid"] = torch.Tensor(
                        np.stack(viz_gt_vid_list, axis=0)
                    )  # batch_size, 3, height, width
                    # batch["viz_vid"] = torch.cat([batch["viz_gt_vid"], batch["viz_pred_vid"]], 3)

                if len(viz_pred_list) > 0:
                    data_batch["viz_gen"] = torch.Tensor(
                        np.stack(viz_pred_list, axis=0)
                    )  # batch_size, 3, height, width
                    data_batch["viz_gen_vid"] = torch.Tensor(
                        np.stack(viz_pred_vid_list, axis=0)
                    )  # batch_size, 3, height, width

                # Move back to training mode if initially training
                if is_training:
                    self.neural_network.train()

        except Exception as e:
            logging.error(f"Error in postprocessing after optimization: {e}")
            raise

        return data_batch

    def _visualize_graph(
        self,
        node_features,
        edge_features,
        fig_size=(3, 3),
        title="",
        horizon=True,
        cam_dist=4.0,
        n_frames=None,
        moving_eid=None,
        gt_mesh_list=None,
    ):
        """
        This function visualizes a graph structure based on provided node and edge features.
        It supports rendering the graph topology, and if provided, it can also compare against ground truth mesh data.
        The function allows for creating animations by specifying the number of frames and can highlight movement in particular edges.
        The visualization can be customized in terms of size, camera distance, and title.
        """
        cat_dim = 1 if horizon else 0

        # Get graph object from node and edge feature vectors
        G = get_G_from_VE(
            node_features,
            edge_features,
            configuration=self.ablations,
            feature_size=self.feature_size,
            feature_index=self.feature_index,
            body_categories=self.body_categories,
            asset_categories=self.asset_categories,
        )

        # Get the meshes from the nodes of a graph object
        G = self._extract_mesh_for_G(G, gt_mesh_list)

        # Visualize the graph topology
        _visualize_graph = viz_G_topology(
            G,
            title=title,
            show_border=False,
            use_categorical_asset=self.ablations["use_categorical_asset"],
            use_categorical_body=self.ablations["use_categorical_body"],
            use_categorical_joint=self.ablations["use_categorical_joint"],
        )

        # Rendering options
        render_shape = (self.viz_dpi * fig_size[0], self.viz_dpi * fig_size[1])
        if n_frames is None:
            n_frames = self.viz_frame_N
        if moving_eid is not None:
            assert isinstance(moving_eid, int)
            moving_mask = {}
            for cnt, e in enumerate(G.edges):
                if cnt == moving_eid:
                    moving_mask[e] = True
                else:
                    moving_mask[e] = False
        else:
            moving_mask = None

        # Visualize articulated object
        render_list = viz_G_BB(
            G,
            shape=render_shape,
            cam_dist=cam_dist,
            viz_frame_N=n_frames,
            moving_mask=moving_mask,
            use_categorical_joint=self.ablations["use_categorical_joint"],
        )

        render_list = [
            np.concatenate([_visualize_graph, gif], axis=cat_dim) for gif in render_list
        ]
        if gt_mesh_list is not None:
            gt_render_list = viz_G_BB(
                G,
                shape=render_shape,
                cam_dist=cam_dist,
                viz_frame_N=n_frames,
                moving_mask=moving_mask,
                render_flags=1024,  # skip back cull
                use_categorical_joint=self.ablations["use_categorical_joint"],
            )
            render_list = [
                np.concatenate([f1, f2], axis=cat_dim)
                for f1, f2 in zip(render_list, gt_render_list)
            ]

        return _visualize_graph, render_list


class NeuralArticulationDDM(torch.nn.Module):
    """
    An implementation of Denoising Diffusion Model (DDM) over a graph transformer network
    """

    def __init__(self, config) -> None:
        """
        Initializes the NeuralArticulationDDM network with the given configuration.

        Args:
            config (Dict): Configuration dictionary for the network.
        """
        super().__init__()

        self.network_dict = torch.nn.ModuleDict()
        self.config = deepcopy(config)
        self.max_nodes = self.config["dataset"].get("max_nodes", 8)

        # Initializa ablation parameters
        self._init_ablation_parameters()

        # Initialize the image decoder
        if self.ablations["use_node_2D_preencoded_latent"]:
            pass
            # self._init_image_decoder()

        # Initialize denoising diffusion parameters
        self._init_diffusion()

        # Initialize the graph denoising network used in the diffusion process
        self._init_graph_denoiser()

        # Activate/deactivate loss weighting
        if "training" in self.config:
            self.use_separate_loss = self.config["training"].get(
                "use_separate_loss", False
            )
        else:
            self.use_separate_loss = False

        # Mask
        self.use_hard_node_mask = self.config["structure_generator"].get(
            "use_hard_node_mask", False
        )
        if self.use_hard_node_mask:
            logging.warning("use_hard_node_mask is True")

        return

    def _init_ablation_parameters(self) -> None:
        # Ablations
        self.ablations = {
            "use_categorical_asset": self.config.get("use_categorical_asset", False),
            "use_categorical_body": self.config.get("use_categorical_body", False),
            "use_categorical_joint": self.config.get("use_categorical_joint", False),
            "use_node_orientation": self.config.get("use_node_orientation", False),
            "use_manifold_plucker": self.config.get("use_manifold_plucker", False),
            "use_node_2D_preencoded_latent": self.config.get(
                "use_node_2D_preencoded_latent", False
            ),
        }

        # Dimensionalities
        dimensions = self.config["dataset"]["dimensions"]
        self.feature_size = {
            "node_oc": dimensions.get("d_node_existence", 1),
            "node_sa": dimensions.get("d_node_semantic_label_asset", 46),
            "node_sb": dimensions.get("d_node_semantic_label_body", 107),
            "node_bb": dimensions.get("d_node_bounding_box", 3),
            "node_r": dimensions.get("d_node_orientation", 3),
            "node_t": dimensions.get("d_node_position", 3),
            "node_2d": dimensions.get("d_node_2d_latent", 64),
            "edge_c": dimensions.get("d_edge_chirality", 3),
            "edge_j": dimensions.get("d_edge_category", 3),
            "edge_p": dimensions.get("d_edge_plucker", 6),
            "edge_m": dimensions.get("d_edge_plucker_manifold", 5),
            "edge_l": dimensions.get("d_edge_joint_limit", 4),
        }

        node_sa_increment = (
            self.feature_size["node_sa"]
            if self.ablations["use_categorical_asset"]
            else 0
        )
        node_sb_increment = (
            self.feature_size["node_sb"]
            if self.ablations["use_categorical_body"]
            else 0
        )
        node_r_increment = (
            self.feature_size["node_r"] if self.ablations["use_node_orientation"] else 0
        )
        node_2d_increment = (
            self.feature_size["node_2d"]
            if self.ablations["use_node_2D_preencoded_latent"]
            else 0
        )
        edge_j_increment = (
            self.feature_size["edge_j"]
            if self.ablations["use_categorical_joint"]
            else 0
        )
        edge_pm_increment = (
            self.feature_size["edge_m"]
            if self.ablations["use_manifold_plucker"]
            else self.feature_size["edge_p"]
        )

        # Based on the ablation options, define the dimensionality of the parameter vectors
        self.feature_size["node"] = (
            self.feature_size["node_oc"]
            + node_sa_increment
            + node_sb_increment
            + self.feature_size["node_bb"]
            + node_r_increment
            + self.feature_size["node_t"]
            + node_2d_increment
        )

        self.feature_size["edge"] = (
            self.feature_size["edge_c"]
            + edge_j_increment
            + edge_pm_increment
            + self.feature_size["edge_l"]
        )

        # Compute the starting index of every feature
        self.feature_index = {
            "node_oc": 0,
            "node_sa": self.feature_size["node_oc"],
            "node_sb": self.feature_size["node_oc"] + node_sa_increment,
            "node_bb": self.feature_size["node_oc"]
            + node_sa_increment
            + node_sb_increment,
            "node_r": self.feature_size["node_oc"]
            + node_sa_increment
            + node_sb_increment
            + self.feature_size["node_bb"],
            "node_t": self.feature_size["node_oc"]
            + node_sa_increment
            + node_sb_increment
            + self.feature_size["node_bb"]
            + node_r_increment,
            "node_2d": self.feature_size["node_oc"]
            + node_sa_increment
            + node_sb_increment
            + self.feature_size["node_bb"]
            + node_r_increment
            + self.feature_size["node_t"],
            "edge_c": 0,
            "edge_j": self.feature_size["edge_c"],
            "edge_p": self.feature_size["edge_c"] + edge_j_increment,
            "edge_m": self.feature_size["edge_c"] + edge_j_increment,
            "edge_l": self.feature_size["edge_c"]
            + edge_j_increment
            + edge_pm_increment,
        }

        self.node_flags = [
            ("oc", True, self.feature_index["node_oc"], self.feature_size["node_oc"]),
            (
                "sa",
                self.ablations["use_categorical_asset"],
                self.feature_index["node_sa"],
                self.feature_size["node_sa"],
            ),
            (
                "sb",
                self.ablations["use_categorical_body"],
                self.feature_index["node_sb"],
                self.feature_size["node_sb"],
            ),
            ("bb", True, self.feature_index["node_bb"], self.feature_size["node_bb"]),
            (
                "r",
                self.ablations["use_node_orientation"],
                self.feature_index["node_r"],
                self.feature_size["node_r"],
            ),
            ("t", True, self.feature_index["node_t"], self.feature_size["node_t"]),
            (
                "2d",
                self.ablations["use_node_2D_preencoded_latent"],
                self.feature_index["node_2d"],
                self.feature_size["node_2d"],
            ),
        ]

        self.edge_flags = [
            ("c", True, self.feature_index["edge_c"], self.feature_size["edge_c"]),
            (
                "j",
                self.ablations["use_categorical_joint"],
                self.feature_index["edge_j"],
                self.feature_size["edge_j"],
            ),
            (
                "m",
                self.ablations["use_manifold_plucker"],
                self.feature_index["edge_m"],
                self.feature_size["edge_m"],
            ),
            (
                "p",
                not self.ablations["use_manifold_plucker"],
                self.feature_index["edge_p"],
                self.feature_size["edge_p"],
            ),
            ("l", True, self.feature_index["edge_l"], self.feature_size["edge_l"]),
        ]

    def _init_diffusion(self) -> None:
        """
        Initialize the Diffusion model.
        """
        # 1000 diffusion time steps according to the NAP paper
        self.diffusion_time_steps = self.config["structure_generator"][
            "denoising_diffusion"
        ].get("time_steps", 1000)

        # Beta scheduling
        schedule_type = self.config["structure_generator"]["denoising_diffusion"].get(
            "schedule_type", "linear"
        )
        if schedule_type != "cosine":
            beta_start = self.config["structure_generator"]["denoising_diffusion"].get(
                "beta_start", 0.001
            )
            beta_end = self.config["structure_generator"]["denoising_diffusion"].get(
                "beta_end", 0.02
            )

        if schedule_type == "linear":
            beta = np.linspace(
                beta_start, beta_end, self.diffusion_time_steps, dtype=np.float32
            )
        elif schedule_type == "warm0.1":
            beta = beta_end * np.ones(self.diffusion_time_steps, dtype=np.float32)
            warmup_time = int(self.diffusion_time_steps * 0.1)
            beta[:warmup_time] = np.linspace(
                beta_start, beta_end, warmup_time, dtype=np.float32
            )
        elif schedule_type == "warm0.2":
            beta = beta_end * np.ones(self.diffusion_time_steps, dtype=np.float32)
            warmup_time = int(self.diffusion_time_steps * 0.2)
            beta[:warmup_time] = np.linspace(
                beta_start, beta_end, warmup_time, dtype=np.float32
            )
        elif schedule_type == "warm0.5":
            beta = beta_end * np.ones(self.diffusion_time_steps, dtype=np.float32)
            warmup_time = int(self.diffusion_time_steps * 0.5)
            beta[:warmup_time] = np.linspace(
                beta_start, beta_end, warmup_time, dtype=np.float32
            )
        elif schedule_type == "cosine":

            def beta_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
                """
                Create a beta schedule that discretizes the given alpha_t_bar function,
                which defines the cumulative product of (1-beta) over time from t = [0,1].
                :param num_diffusion_timesteps: the number of betas to produce.
                :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                                produces the cumulative product of (1-beta) up to that
                                part of the diffusion process.
                :param max_beta: the maximum beta to use; use values lower than 1 to
                                prevent singularities.
                """
                beta = []
                for i in range(num_diffusion_timesteps):
                    t1 = i / num_diffusion_timesteps
                    t2 = (i + 1) / num_diffusion_timesteps
                    beta.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

                return np.array(beta)

            beta = beta_for_alpha_bar(
                self.diffusion_time_steps,
                lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(schedule_type)

        self.beta = torch.from_numpy(beta).float()
        self.alpha = torch.from_numpy(np.ones_like(beta) - beta).float()
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Buffers are tensors that are not to be considered model parameters.
        # That is, they are not trainable and do not get updated during backpropagation.
        # However, unlike regular tensors, buffers are part of the module's state.
        # This means they are included in the module's state dictionary (state_dict) and
        # are moved along with the module to the specified device (like GPU) or saved along with the module.
        self.register_buffer("betas", self.beta)
        self.register_buffer("alphas", self.alpha)
        self.register_buffer("alpha_bars", self.alpha_bar)

        # Number of randomly sampled time steps.
        if "training" in self.config:
            self.N_t_training = self.config["training"].get("N_t_training", 1)
        else:
            self.N_t_training = 1

    def _init_image_decoder(self) -> None:
        """
        Initialize the image decoder
        """
        image_decoder_config = self.config["structure_generator"]["image_autoencoder"]

        # Shape decoder network
        self.input_image_hw = image_decoder_config["ddconfig"].get("resolution", 256)
        self.output_channels = image_decoder_config.get("out_ch", 3)  # rgb
        self.output_image_hw = image_decoder_config.get("output_hw", 137)
        self.vqvae = ImageVQVAE(self.config)
        ckpt_fn = os.path.join(
            self.config["base_directory"],
            self.config["output_directory"],
            image_decoder_config.get("checkpoint_path", ""),
        )
        ckpt = torch.load(ckpt_fn, map_location="cpu")
        self.vqvae.model_resume(ckpt, is_initialization=True, network_name=["all"])
        logging.info("Loaded image decoder weights from %s", ckpt_fn)

        # Copy model to GPU memory and prepare for evaluation
        self.vqvae.to_gpus()

        # Set in evaluation mode to save memory
        self.vqvae.set_eval()

    def _init_graph_denoiser(self) -> None:
        """
        Initialize the denoising network
        """

        # Conditionally construct node_feature list based on configuration flags
        node_feature = [
            self.feature_size["node_oc"],  # Always included
            *(
                [self.feature_size["node_sa"]]
                if self.ablations["use_categorical_asset"]
                else []
            ),
            *(
                [self.feature_size["node_sb"]]
                if self.ablations["use_categorical_body"]
                else []
            ),
            self.feature_size["node_bb"],  # Always included
            *(
                [self.feature_size["node_r"]]
                if self.ablations["use_node_orientation"]
                else []
            ),
            self.feature_size["node_t"],  # Always included
            *(
                [self.feature_size["node_2d"]]
                if self.ablations["use_node_2D_preencoded_latent"]
                else []
            ),
        ]
        edge_feature = [
            self.feature_size["edge_c"],  # Always included
            *(
                [self.feature_size["edge_j"]]
                if self.ablations["use_categorical_joint"]
                else []
            ),
            *(
                [self.feature_size["edge_m"]]
                if self.ablations["use_manifold_plucker"]
                else [self.feature_size["edge_p"]]
            ),
            self.feature_size["edge_l"],  # Always included
        ]

        # Graph ATtention (GAT) denoiser:
        denoiser_config = self.config["structure_generator"]["graph_transformer"]
        self.network_dict["denoiser"] = GraphTransformer(
            node_features_struct=node_feature,
            edge_features_struct=edge_feature,
            hidden_node_features_dim=denoiser_config.get(
                "hidden_node_feature_dim", [512, 512, 512, 512, 512, 512]
            ),
            hidden_edge_features_dim=denoiser_config.get(
                "hidden_edge_feature_dim", [512, 512, 512, 512, 512, 512]
            ),
            out_node_features_dim=denoiser_config.get(
                "out_node_feature_dim", [256, 256, 256]
            ),
            out_edge_features_dim=denoiser_config.get(
                "out_edge_feature_dim", [256, 128, 64]
            ),
            p_emb_dim=denoiser_config.get("p_emb_dim", 200),
            t_emb_dim=denoiser_config.get("t_emb_dim", 100),
            num_attention_heads=denoiser_config.get("num_gat_heads", 32),
            max_nodes=self.max_nodes,
            diffusion_time_steps=self.diffusion_time_steps,
            use_batch_normalization=denoiser_config.get(
                "use_batch_normalization", False
            ),
            type_graph_layers=denoiser_config.get("type_graph_layers", "gt"),
            dir_handling=denoiser_config.get("dir_handling", True),
            sym_sync=denoiser_config.get("sym_sync", False),
        )

    @torch.no_grad()  # Ensure that the operation does not track gradients: useful for inference or evaluation phases.
    def _extract_tree_from_min_span_tree_matrix(
        self, min_span_tree_matrix, node_mask
    ) -> torch.Tensor:
        """
        Extracts a minimum spanning tree for each sample in a batch.

        Parameters:
        min_span_tree_matrix (Tensor): A batch of matrices from which to extract minimum spanning trees.
        node_mask (Tensor): A mask indicating the valid nodes for each sample in the batch.

        Returns:
        Tensor: A batch of binary matrices, each representing the extracted minimum spanning tree for a sample.
        """
        device = min_span_tree_matrix.device
        batch_size = min_span_tree_matrix.shape[0]

        # Iterates over each sample in the batch.
        binary_mask_list = []
        for batch_id in range(batch_size):
            _node_mask = node_mask[batch_id] > 0
            sub_matrix = -min_span_tree_matrix[batch_id, _node_mask][
                :, _node_mask
            ]  # minus
            sub_matrix = sub_matrix - sub_matrix.min() + 1.0
            num_nodes = sub_matrix.shape[0]
            sub_matrix = sub_matrix.cpu() * (1.0 - torch.eye(num_nodes))
            Tcsr = minimum_spanning_tree(sub_matrix).toarray()
            Tcsr = (Tcsr > 1e-8).astype(np.float32)

            # Ensures that the tree connects all nodes (tree property).
            assert Tcsr.sum() == num_nodes - 1

            # Reconstructs the full matrix from the sub-matrix.
            full_matrix = np.zeros((self.max_nodes * self.max_nodes))
            _matrix_mask = _node_mask[:, None] * _node_mask[None, :]
            full_matrix[_matrix_mask.cpu().numpy().reshape(-1)] = Tcsr.reshape(-1)
            full_matrix = full_matrix.reshape(self.max_nodes, self.max_nodes)
            binary_mask_list.append(torch.from_numpy(full_matrix).float())

        # Sometimes, the Tcsr will set lower triangle to 1
        binary_mask_list = torch.stack(binary_mask_list, 0).to(device)
        binary_mask_list = binary_mask_list + binary_mask_list.permute(0, 2, 1)
        binary_mask_list = binary_mask_list > 0

        return binary_mask_list.float()

    def _manifold_to_plucker(self, plucker_manifold) -> torch.Tensor:
        phi = plucker_manifold[..., 0]
        theta = plucker_manifold[..., 1]
        n = plucker_manifold[..., 2:]
        l = torch.stack(
            [
                torch.cos(phi) * torch.sin(theta),
                torch.sin(phi) * torch.sin(theta),
                torch.cos(theta),
            ],
            dim=-1,  # Stack along the last dimension
        )
        m = torch.linalg.cross(n, l)
        return torch.cat([l, m], -1)

    def _project_to_plucker(self, edge_features) -> torch.Tensor:
        """
        Make sure that the denoised edge features corresponding to the Plücker joint parametrization comply
        with the structure of actual Plücker coordinates by "projecting" them into the "nearest" valid coordinates.
        """

        # Sanity check
        assert (
            self.ablations["use_manifold_plucker"] == False
        ), "The manifold plucker representation is incompatible with the use of the projection step."

        # edge_features: [chirality, label (if use_categorical_joint), plucker, rlim, plim]
        idx_pre = self.feature_index["edge_p"]
        idx_mid = idx_pre + 3
        idx_post = self.feature_index["edge_l"]
        l_pred, m_pred = (
            edge_features[..., idx_pre:idx_mid],
            edge_features[..., idx_mid:idx_post],
        )
        l_pred = torch.nn.functional.normalize(l_pred, dim=-1)
        _inner = (l_pred * m_pred).sum(-1, keepdim=True)
        m_pred = m_pred - l_pred * _inner
        edge_features = torch.cat(
            [
                edge_features[..., :idx_pre],
                l_pred,
                m_pred,
                edge_features[..., idx_post:],
            ],
            -1,
        )

        return edge_features

    def _make_sure_two_fg_nodes(self, node_mask) -> torch.Tensor:
        """
        Ensure that at least two nodes per batch item are selected.

        Parameters:
        node_mask (Tensor): A 2D tensor with shape (batch_size, num_nodes) representing the mask for nodes.
        """
        # Selects the indices of the top 2 values in node_mask along the last dimension (num_nodes).
        top2ind = torch.topk(node_mask, 2, dim=-1)[1]

        # Convert node_mask to a boolean tensor node_mask: batch_size, num_nodes
        node_mask = torch.scatter(
            input=node_mask, dim=1, index=top2ind, src=torch.ones_like(top2ind).float()
        )
        node_mask = node_mask > 0.5

        # Assert that for each item in the batch, at least two nodes are selected (i.e., sum along the last dimension is >= 2).
        assert node_mask.sum(-1).min() >= 2

        return node_mask.float()

    @torch.no_grad()  # Ensure that the operation does not track gradients: useful for inference or evaluation phases.
    def generate(
        self,
        noisy_node_features,
        noisy_edge_features,
        node_scale,
        edge_scale,
        node_mask=None,
        known_node_features=None,
        temperature=1.0,
        eta=0.1,  # eta = 0 # Deterministic
    ) -> None:
        """
        Generate data in an unconditioned manner using the model's denoising network.

        Parameters:
        - noisy_node_features: Tensor of shape [batch_size, num_nodes, feature_dim], initial noisy node features.
        - noisy_edge_features: Tensor of shape [batch_size, num_edge_features, feature_dim], initial noisy edge features.
        - node_scale: Scalar or Tensor for scaling node features after generation.
        - edge_scale: Scalar or Tensor for scaling edge features after generation.
        - node_mask: Optional Tensor indicating the valid nodes in the batch.
        - known_node_features: Optional Tensor of known node features for conditional generation.

        Returns:
        - node_features: Tensor containing generated node features.
        - ret_edge: Tensor containing generated edge features and types.
        """
        # node_mask is used as padded batch, the number of nodes is decided during sampling
        # noisy_node_features: [batch_size,num_nodes,1+6+C_shapecode]; noisy_edge_features: [batch_size, num_nodes(num_nodes-1)/2,13]; node_mask: [batch_size,num_nodes,1]
        # shapecode_std: C_shapecode

        # Preliminary checks
        batch_size, num_nodes, _ = noisy_node_features.shape
        # _, num_edges, _ = noisy_edge_features.shape
        assert (
            noisy_node_features.shape[1] == self.max_nodes
        ), "Currently supports only max_nodes for inference."
        assert noisy_edge_features.shape[:2] == (
            batch_size,
            num_nodes * (num_nodes - 1) // 2,
        )

        # Initialize flags and clone inputs
        node_condition_flag = known_node_features is not None
        if node_condition_flag:
            logging.info("Experimental: conditional on node_features")
            assert (
                known_node_features.shape == noisy_node_features.shape
            ), f"known_node_features shape {known_node_features.shape} != noise {noisy_node_features.shape}"
        else:
            known_node_features = torch.zeros_like(noisy_node_features)

        if node_mask is not None:
            assert (
                noisy_node_features[..., 0] == node_mask
            ).all(), "Node mask mismatch."

        edge_features = noisy_edge_features.clone()
        node_features = noisy_node_features.clone()

        logging.info(f"Diffusion Generation with {self.diffusion_time_steps} steps...")

        # Main diffusion loop
        for t in tqdm(
            list(range(self.diffusion_time_steps))[::-1], desc="Diffusion Steps"
        ):
            t_pad = torch.full(
                (edge_features.shape[0],),
                t,
                dtype=torch.long,
                device=edge_features.device,
            )
            t_pad_1 = torch.full(
                (edge_features.shape[0],),
                max(t - 1, 0),
                dtype=torch.long,
                device=edge_features.device,
            )
            alpha_t = self.alphas[t_pad][:, None, None]
            alpha_t_bar = self.alpha_bars[t_pad][:, None, None]
            beta_t = self.betas[t_pad][:, None, None]

            # Execute denoising step:
            eps_node, eps_edge = self.network_dict["denoiser"](
                node_features, edge_features, t_pad, node_mask=node_mask
            )

            # Denoising step
            explicit = False
            if explicit:  # Denoising diffusion sampling
                sigma_t = beta_t.sqrt()

                # Conditional node features handling
                if node_condition_flag:
                    if t > 0:
                        node_features = alpha_t_bar.sqrt() * known_node_features + (
                            1.0 - alpha_t_bar
                        ).sqrt() * torch.randn_like(node_features)
                    else:
                        node_features = known_node_features
                else:
                    node_features = (
                        node_features
                        - ((1.0 - alpha_t) / (1.0 - alpha_t_bar).sqrt()) * eps_node
                    ) / alpha_t.sqrt()
                    if t > 0:
                        node_features += sigma_t * torch.randn_like(node_features)

                if self.use_hard_node_mask and node_mask is not None:
                    node_features[..., 0] = node_mask

                # Edge features handling
                edge_features = (
                    edge_features
                    - ((1.0 - alpha_t) / (1.0 - alpha_t_bar).sqrt()) * eps_edge
                ) / alpha_t.sqrt()
                if t > 0:
                    edge_features += sigma_t * torch.randn_like(edge_features)

            else:  # DDIM sampling
                if t > 0:
                    alpha_t_prev = self.alphas[t_pad_1][:, None, None]
                else:
                    alpha_t_prev = torch.ones_like(alpha_t)
                sigma_t = eta * torch.sqrt(
                    (1.0 - alpha_t_prev)
                    * (1.0 - alpha_t / alpha_t_prev)
                    / (1.0 - alpha_t)
                )

                # Conditional node features handling
                if node_condition_flag:
                    node_features = alpha_t_bar.sqrt() * known_node_features + (
                        1.0 - alpha_t_bar
                    ).sqrt() * torch.randn_like(node_features)
                else:  # (weird but otherwise does not work !)
                    node_features = (
                        node_features
                        - ((1.0 - alpha_t) / (1.0 - alpha_t_bar).sqrt()) * eps_node
                    ) / alpha_t.sqrt()

                if self.use_hard_node_mask and node_mask is not None:
                    node_features[..., 0] = node_mask

                # Edge features handling (weird but otherwise does not work !)
                edge_features = (
                    edge_features
                    - ((1.0 - alpha_t) / (1.0 - alpha_t_bar).sqrt()) * eps_edge
                ) / alpha_t.sqrt()

                # From DDIM paper https://arxiv.org/pdf/2010.02502.pdf
                # Current prediction for denoised features node_features_0, edge_features_0
                pred_node_features = (
                    node_features - (1.0 - alpha_t).sqrt() * eps_node
                ) / alpha_t.sqrt()
                pred_edge_features = (
                    edge_features - (1.0 - alpha_t).sqrt() * eps_edge
                ) / alpha_t.sqrt()

                # Direction pointing to posterior
                dir_node_features = (1.0 - alpha_t_prev - sigma_t**2).sqrt() * eps_node
                dir_edge_features = (1.0 - alpha_t_prev - sigma_t**2).sqrt() * eps_edge

                # Noise
                noise_node_features = (
                    sigma_t * torch.randn_like(node_features) * temperature
                )
                noise_edge_features = (
                    sigma_t * torch.randn_like(edge_features) * temperature
                )

                # Update node_features
                node_features = alpha_t_prev.sqrt() * pred_node_features
                node_features += dir_node_features
                node_features += noise_node_features

                # Update edge_features
                edge_features = alpha_t_prev.sqrt() * pred_edge_features
                edge_features += dir_edge_features
                edge_features += noise_edge_features

        # Post-processing steps: normalization, projection, and edge type extraction
        if not self.ablations["use_node_orientation"]:
            # Assume R0 = I, and append this to the node_features
            # o_i(1), sa_i(46) + sb_i(107) + bb_i(3) = 157
            idx_r = self.feature_index["node_r"]
            node_features = torch.cat(
                [
                    node_features[..., :idx_r],
                    torch.zeros_like(node_features[..., 1:4]),
                    node_features[..., idx_r:],
                ],
                -1,
            )
        node_features = node_features * node_scale[None, None, :]
        edge_features = edge_features * edge_scale[None, None, :]
        confirmed_node_mask = self._make_sure_two_fg_nodes(node_features[..., 0])
        node_features[..., 0] = confirmed_node_mask

        # Extract the sparse asset graph using Minimum Spaning Tree (MST)
        # post-processing of the complete asset graph.
        edge_type = edge_features[..., : self.feature_size["edge_c"]]
        # how 1,2 edge is larger than the prob to be the empty edge
        E_value = edge_type[..., 1:].max(-1).values - edge_type[..., 0]
        edge_matrix = self.network_dict["denoiser"].scatter_trilist_to_matrix(
            E_value[..., None]
        )
        edge_matrix = edge_matrix.squeeze(-1)
        edge_matrix = self._extract_tree_from_min_span_tree_matrix(
            edge_matrix, confirmed_node_mask
        )
        gather_ind = self.network_dict["denoiser"].tri_ind_to_full_ind.clone()
        edge_matrix = edge_matrix.reshape(batch_size, -1)
        edge_fg = torch.gather(
            edge_matrix,
            1,
            gather_ind[None, :].expand(batch_size, -1).to(edge_matrix.device),
        )
        edge_fg = edge_fg > 0
        final_edge_type = torch.zeros_like(edge_type[..., 0]).long()
        final_edge_type[edge_fg] = edge_type[..., 1:].argmax(-1)[edge_fg] + 1
        final_edge_type = torch.nn.functional.one_hot(
            final_edge_type, num_classes=3
        ).float()

        # Project the plucker to valid plucker
        if not self.ablations["use_manifold_plucker"]:
            edge_features = self._project_to_plucker(edge_features)

        # Pack output
        ret_edge = torch.cat(
            [final_edge_type, edge_features[..., self.feature_index["edge_j"] :]], -1
        )

        return node_features, ret_edge

    @torch.no_grad()  # Ensure that the operation does not track gradients: useful for inference or evaluation phases.
    def masked_generate(
        self,
        noisy_node_features,
        noisy_edge_features,
        node_scale,
        edge_scale,
        node_mask=None,
        known_node_features=None,
        edge_mask=None,
        known_edge_features=None,
        temperature=1.0,
        eta=0.1,  # eta = 0 # Deterministic
    ) -> None:
        """
        Generate data in an unconditioned manner using the model's denoising network.

        Parameters:
        - noisy_node_features: Tensor of shape [batch_size, num_nodes, feature_dim], initial noisy node features.
        - noisy_edge_features: Tensor of shape [batch_size, num_edge_features, feature_dim], initial noisy edge features.
        - node_scale: Scalar or Tensor for scaling node features after generation.
        - edge_scale: Scalar or Tensor for scaling edge features after generation.
        - node_mask: Optional Tensor indicating the valid nodes in the batch.
        - known_node_features: Optional Tensor of known node features for conditional generation.
        - edge_mask: Optional Tensor indicating the valid edges in the batch.
        - known_edge_features: Optional Tensor of known edge features for conditional generation.

        Returns:
        - node_features: Tensor containing generated node features.
        - ret_edge: Tensor containing generated edge features and types.
        """
        # node_mask is used as padded batch, the number of nodes is decided during sampling
        # noisy_node_features: [batch_size,num_nodes,1+6+C_shapecode]; noisy_edge_features: [batch_size, num_nodes(num_nodes-1)/2,13]; node_mask: [batch_size,num_nodes,1]
        # shapecode_std: C_shapecode

        # Preliminary checks
        batch_size, num_nodes, _ = noisy_node_features.shape
        # _, num_edges, _ = noisy_edge_features.shape
        assert (
            noisy_node_features.shape[1] == self.max_nodes
        ), "Currently supports only max_nodes for inference."
        assert noisy_edge_features.shape[:2] == (
            batch_size,
            num_nodes * (num_nodes - 1) // 2,
        )

        # Initialize flags and clone inputs
        node_condition_flag = known_node_features is not None
        if node_condition_flag:
            logging.info("Experimental: conditional on node_features")
            assert (
                known_node_features.shape == noisy_node_features.shape
            ), f"known_node_features shape {known_node_features.shape} != noise {noisy_node_features.shape}"
            assert node_mask is not None
            assert len(node_mask.unique()) <= 2
        else:
            known_node_features = torch.zeros_like(noisy_node_features)
            node_mask = torch.ones_like(noisy_node_features)

        edge_condition_flag = known_edge_features is not None
        if edge_condition_flag:
            logging.info("Experimental: conditional on node_features")
            assert (
                known_edge_features.shape == noisy_edge_features.shape
            ), f"known_edge_features shape {known_edge_features.shape} != noise {noisy_edge_features.shape}"
            assert edge_mask is not None
            assert len(edge_mask.unique()) <= 2
        else:
            known_edge_features = torch.zeros_like(noisy_edge_features)
            edge_mask = torch.ones_like(noisy_edge_features)

        node_features = noisy_node_features.clone()
        edge_features = noisy_edge_features.clone()

        logging.info(f"Diffusion Generation with {self.diffusion_time_steps} steps...")

        # Main diffusion loop
        for t in tqdm(
            list(range(self.diffusion_time_steps))[::-1], desc="Diffusion Steps"
        ):
            t_pad = torch.full(
                (edge_features.shape[0],),
                t,
                dtype=torch.long,
                device=edge_features.device,
            )
            t_pad_1 = torch.full(
                (edge_features.shape[0],),
                max(t - 1, 0),
                dtype=torch.long,
                device=edge_features.device,
            )
            alpha_t = self.alphas[t_pad][:, None, None]
            alpha_t_bar = self.alpha_bars[t_pad][:, None, None]
            beta_t = self.betas[t_pad][:, None, None]

            # Execute denoising step:
            eps_node, eps_edge = self.network_dict["denoiser"](
                node_features, edge_features, t_pad, node_mask=None
            )

            # Denoising step
            explicit = False
            if explicit:  # Denoising diffusion sampling
                sigma_t = beta_t.sqrt()

                # Unconditional proposal
                node_features_uncond = (
                    node_features
                    - ((1.0 - alpha_t) / (1.0 - alpha_t_bar).sqrt()) * eps_node
                ) / alpha_t.sqrt()
                edge_features_uncond = (
                    edge_features
                    - ((1.0 - alpha_t) / (1.0 - alpha_t_bar).sqrt()) * eps_edge
                ) / alpha_t.sqrt()

                # Condition encoding
                node_features_cond = alpha_t_bar.sqrt() * known_node_features + (
                    1.0 - alpha_t_bar
                ).sqrt() * torch.randn_like(node_features)
                edge_features_cond = alpha_t_bar.sqrt() * known_edge_features + (
                    1.0 - alpha_t_bar
                ).sqrt() * torch.randn_like(edge_features)

                ilvr = False
                if ilvr:
                    # Iterative Latent Variable Refinement for Conditioned Generation Without Retraining
                    # https://arxiv.org/pdf/2108.02938
                    # https://pubs.acs.org/doi/epdf/10.1021/acs.jcim.3c00667

                    node_features = node_features_uncond + node_features_cond
                    edge_features = edge_features_uncond + edge_features_cond

                else:
                    # Conditional features handling
                    node_features = (
                        node_mask * node_features_uncond
                        + (1.0 - node_mask) * node_features_cond
                    )
                    edge_features = (
                        edge_mask * edge_features_uncond
                        + (1.0 - edge_mask) * edge_features_cond
                    )

                if t > 0:
                    node_features += (
                        sigma_t * torch.randn_like(node_features)
                    ) * node_mask
                    edge_features += (
                        sigma_t * torch.randn_like(edge_features)
                    ) * edge_mask
                else:  # Force the final state
                    if known_node_features is not None:
                        node_features = (
                            node_mask * node_features
                            + (1.0 - node_mask) * known_node_features
                        )
                    if known_edge_features is not None:
                        edge_features = (
                            edge_mask * edge_features
                            + (1.0 - edge_mask) * known_edge_features
                        )

            else:  # DDIM sampling
                if t > 0:
                    alpha_t_prev = self.alphas[t_pad_1][:, None, None]
                else:
                    alpha_t_prev = torch.ones_like(alpha_t)
                sigma_t = eta * torch.sqrt(
                    (1.0 - alpha_t_prev) * (1.0 - alpha_t / alpha_t_prev) / beta_t
                )

                # KEY FEATURE
                scaling_factor = (1.0 - alpha_t_bar).sqrt()
                node_features = (
                    node_features - (beta_t / scaling_factor) * eps_node
                ) / alpha_t.sqrt()
                edge_features = (
                    edge_features - (beta_t / scaling_factor) * eps_edge
                ) / alpha_t.sqrt()

                # node_features_orig = (alpha_t_bar.sqrt() * known_node_features + scaling_factor * torch.randn_like(node_features))
                # node_features = node_mask * node_features + (1. - node_mask) * node_features_orig
                # edge_features_orig = (alpha_t_bar.sqrt() * known_edge_features + scaling_factor * torch.randn_like(edge_features))
                # edge_features = edge_mask * edge_features + (1. - edge_mask) * edge_features_orig

                node_features = (
                    node_mask * node_features + (1.0 - node_mask) * known_node_features
                )
                edge_features = (
                    edge_mask * edge_features + (1.0 - edge_mask) * known_edge_features
                )

                # From DDIM paper https://arxiv.org/pdf/2010.02502.pdf
                # Current prediction for denoised features node_features_0, edge_features_0
                pred_node_features = (
                    node_features - beta_t.sqrt() * eps_node
                ) / alpha_t.sqrt()
                pred_edge_features = (
                    edge_features - beta_t.sqrt() * eps_edge
                ) / alpha_t.sqrt()

                # Direction pointing to posterior
                dir_node_features = (1.0 - alpha_t_prev - sigma_t**2).sqrt() * eps_node
                dir_edge_features = (1.0 - alpha_t_prev - sigma_t**2).sqrt() * eps_edge

                # Noise
                noise_node_features = (
                    sigma_t * torch.randn_like(node_features) * temperature
                )
                noise_edge_features = (
                    sigma_t * torch.randn_like(edge_features) * temperature
                )

                # Update node_features
                node_features = alpha_t_prev.sqrt() * pred_node_features
                node_features += dir_node_features
                node_features += noise_node_features

                # Update edge_features
                edge_features = alpha_t_prev.sqrt() * pred_edge_features
                edge_features += dir_edge_features
                edge_features += noise_edge_features

        node_features = (
            node_mask * node_features + (1.0 - node_mask) * known_node_features
        )
        edge_features = (
            edge_mask * edge_features + (1.0 - edge_mask) * known_edge_features
        )

        # Post-processing steps: normalization, projection, and edge type extraction
        if not self.ablations["use_node_orientation"]:
            # Assume R0 = I, and append this to the node_features
            # o_i(1), sa_i(46) + sb_i(107) + bb_i(3) = 157
            idx_r = self.feature_index["node_r"]
            node_features = torch.cat(
                [
                    node_features[..., :idx_r],
                    torch.zeros_like(node_features[..., 1:4]),
                    node_features[..., idx_r:],
                ],
                -1,
            )
        node_features = node_features * node_scale[None, None, :]
        edge_features = edge_features * edge_scale[None, None, :]
        confirmed_node_mask = self._make_sure_two_fg_nodes(node_features[..., 0])
        node_features[..., 0] = confirmed_node_mask

        # Extract the sparse asset graph using Minimum Spaning Tree (MST)
        # post-processing of the complete asset graph.
        edge_type = edge_features[..., : self.feature_size["edge_c"]]
        # how 1,2 edge is larger than the prob to be the empty edge
        E_value = edge_type[..., 1:].max(-1).values - edge_type[..., 0]
        edge_matrix = self.network_dict["denoiser"].scatter_trilist_to_matrix(
            E_value[..., None]
        )
        edge_matrix = edge_matrix.squeeze(-1)
        edge_matrix = self._extract_tree_from_min_span_tree_matrix(
            edge_matrix, confirmed_node_mask
        )
        gather_ind = self.network_dict["denoiser"].tri_ind_to_full_ind.clone()
        edge_matrix = edge_matrix.reshape(batch_size, -1)
        edge_fg = torch.gather(
            edge_matrix,
            1,
            gather_ind[None, :].expand(batch_size, -1).to(edge_matrix.device),
        )
        edge_fg = edge_fg > 0
        final_edge_type = torch.zeros_like(edge_type[..., 0]).long()
        final_edge_type[edge_fg] = edge_type[..., 1:].argmax(-1)[edge_fg] + 1
        final_edge_type = torch.nn.functional.one_hot(
            final_edge_type, num_classes=3
        ).float()

        # Project the plucker to valid plucker
        if not self.ablations["use_manifold_plucker"]:
            edge_features = self._project_to_plucker(edge_features)

        # Pack output
        ret_edge = torch.cat(
            [final_edge_type, edge_features[..., self.feature_index["edge_j"] :]], -1
        )

        return node_features, ret_edge

    def _apply_random_mask(self, features, category_ranges):
        """
        Apply random mask to the node features tensor using start indices and ranges.

        Parameters:
        - node_features (torch.Tensor): Tensor of shape [batch_size, num_nodes, num_node_features].
        - category_ranges (list of tuples): Each tuple contains (start_index, range) for a category.
        - mask_prob (float): Probability of masking a feature.

        Returns:
        - masked_node_features (torch.Tensor): Tensor with the same shape as input with masked features.
        """
        batch_size, num_nodes, num_features = features.shape
        masked_features = features.clone()

        for _, existance, start_index, category_range in category_ranges:
            if existance:
                end_index = start_index + category_range

                # Generate a mask that decides whether to zero out the whole category
                category_mask = (
                    torch.rand(batch_size, num_nodes, 1, device=features.device) < 0.5
                ).float()

                # Expand the mask to the size of the category range
                category_mask = category_mask.expand(-1, -1, category_range)

                # Apply the mask to the respective category indices
                masked_features[:, :, start_index:end_index] *= category_mask

        return masked_features

    def forward(self, data_batch, visualize):
        """
        Performs forward and backward diffusion processes.

        Args:
            data_batch (dict): A batch of data containing node and edge features, scales, and potentially other information.
            visualize (bool): A flag indicating whether to include additional visualization data in the output.

        Returns:
            dict: A dictionary containing various output information, including losses, scaled features, and optionally, ground truth data for visualization.

        The method processes the input data through a diffusion model, computes errors and losses, and returns these along with additional information for analysis and visualization.
        """
        output = {}
        output["visualize"] = visualize

        # node_features: [o_i(1), asset_label(46), part_label(107), bbox(3), r_gl(3), t_gl(3)]
        # edge_features: [chirality(3), joint_label(3), plucker(6), rlim(2), plim(2)] if self.ablations["use_categorical_joint"]
        # [chirality(3), plucker(6), rlim(2), plim(2)] else
        node_features_gt = data_batch["V"]
        edge_features_gt = data_batch["E"]

        if not self.ablations["use_node_orientation"]:
            # Remove the node rotation for now: assume that all the mesh rotations are set to 0.
            # o_i(1), sa_i(46) + sb_i(107) + bb_i(3) = 157
            node_features_gt = torch.cat(
                [
                    node_features_gt[..., : self.feature_index["node_r"]],
                    node_features_gt[
                        ...,
                        self.feature_index["node_r"] + self.feature_size["node_r"] :,
                    ],
                ],
                -1,
            )  # Now node_features: [o_i(1), asset_label(46), part_label(107), bbox(3), t_gl(3)]

        batch_size, num_nodes, _ = node_features_gt.shape
        _, num_edges, _ = edge_features_gt.shape

        # Node and edge feature scales
        node_scale = data_batch["V_scale"][0]
        edge_scale = data_batch["E_scale"][0]

        # Can supervise multiple t step for one object in the batch
        t = np.random.randint(
            0, self.diffusion_time_steps, (batch_size * self.N_t_training)
        )
        node_features_0 = node_features_gt[:, None, ...].expand(
            -1, self.N_t_training, -1, -1
        )  # batch_size,N_t_training,num_nodes,node_features
        node_features_0 = node_features_0.reshape(
            batch_size * self.N_t_training, num_nodes, -1
        )  # batch_size*N_t_training,num_nodes,node_features
        edge_features_0 = edge_features_gt[:, None, ...].expand(
            -1, self.N_t_training, -1, -1
        )  # batch_size,N_t_training,num_nodes,edge_features
        edge_features_0 = edge_features_0.reshape(
            batch_size * self.N_t_training, num_nodes * (num_nodes - 1) // 2, -1
        )  # batch_size*N_t_training,num_nodes*(num_nodes-1)/2,edge_features

        # Condition signal
        nodes_features_cond = self._apply_random_mask(node_features_0, self.node_flags)
        edges_features_cond = self._apply_random_mask(edge_features_0, self.edge_flags)

        # Forward diffusion process
        eps_node = torch.randn_like(node_features_0)
        eps_edge = torch.randn_like(edge_features_0)
        alpha_bar_t = self.alpha_bars[t]

        noisy_node_features = (
            alpha_bar_t.sqrt()[:, None, None] * node_features_0
            + (1.0 - alpha_bar_t).sqrt()[:, None, None] * eps_node
        )  # batch_size*N_t_training,num_nodes,node_features
        noisy_edge_features = (
            alpha_bar_t.sqrt()[:, None, None] * edge_features_0
            + (1.0 - alpha_bar_t).sqrt()[:, None, None] * eps_edge
        )  # batch_size*N_t_training,num_nodes*(num_nodes-1)/2,edge_features

        if self.use_hard_node_mask:
            node_mask_gt = node_features_0[..., 0]
            noisy_node_features[..., 0] = (
                node_mask_gt  # also always set the noisy mask to gt
            )
        else:
            node_mask_gt = None

        # Backward diffusion
        eps_node_hat, eps_edge_hat = self.network_dict["denoiser"](
            noisy_node_features,
            noisy_edge_features,
            t,
            nodes_features_cond=nodes_features_cond,
            edges_features_cond=edges_features_cond,
            node_mask=node_mask_gt,
        )

        # Compute the loss
        output = self._compute_losses(
            output, eps_node, eps_node_hat, eps_edge, eps_edge_hat, node_mask_gt
        )

        if visualize:
            output["V_gt"] = data_batch["V"].clone() * node_scale[None, None, :]
            output["E_gt"] = data_batch["E"].clone() * edge_scale[None, None, :]

        output["V_scale"] = node_scale.detach()
        output["E_scale"] = edge_scale.detach()

        return output

    def _compute_losses(
        self, output, eps_node, eps_node_hat, eps_edge, eps_edge_hat, node_mask_gt
    ):
        if self.ablations["use_manifold_plucker"]:
            idx_pre = self.feature_index["edge_m"]
            idx_post_manifold = self.feature_index["edge_l"]
            eps_manifold = eps_edge[..., idx_pre:idx_post_manifold]
            eps_manifold_hat = eps_edge_hat[..., idx_pre:idx_post_manifold]
            eps_plucker = self._manifold_to_plucker(eps_manifold)
            eps_plucker_hat = self._manifold_to_plucker(eps_manifold_hat)

        # Compute the raw losses
        old_loss = False
        if old_loss:
            error_node = (eps_node - eps_node_hat) ** 2
            error_edge = (eps_edge - eps_edge_hat) ** 2

            if self.use_hard_node_mask:
                # Prepare valid edge_features mask
                edge_mask_gt = self.network_dict["denoiser"].get_edge_mask(node_mask_gt)
                loss_node_i = (error_node * node_mask_gt[..., None]).sum(
                    1
                ) / node_mask_gt[..., None].sum(1)
                loss_edge_i = (error_edge * edge_mask_gt[..., None]).sum(
                    1
                ) / edge_mask_gt[..., None].sum(1)
                if self.ablations["use_manifold_plucker"]:
                    loss_plucker = torch.nn.functional.mse_loss(
                        eps_plucker, eps_plucker_hat, reduction="none"
                    )
                    loss_plucker = (loss_plucker * edge_mask_expanded).sum(
                        1
                    ) / edge_mask_expanded.sum(1)
            else:
                loss_node_i = error_node.mean(1)
                loss_edge_i = error_edge.mean(1)
                if self.ablations["use_manifold_plucker"]:
                    loss_plucker = torch.nn.functional.mse_loss(
                        eps_plucker, eps_plucker_hat, reduction="none"
                    ).mean(1)
        else:
            if self.use_hard_node_mask:
                eps_node_hat[..., 0] = eps_node[..., 0]
                node_mask_expanded = node_mask_gt[..., None]
                edge_mask_gt = self.network_dict["denoiser"].get_edge_mask(node_mask_gt)
                edge_mask_expanded = edge_mask_gt[..., None]

                loss_node_i = torch.nn.functional.mse_loss(
                    eps_node, eps_node_hat, reduction="none"
                )
                loss_node_i = (loss_node_i * node_mask_expanded).sum(
                    1
                ) / node_mask_expanded.sum(1)

                loss_edge_i = torch.nn.functional.mse_loss(
                    eps_edge, eps_edge_hat, reduction="none"
                )
                loss_edge_i = (loss_edge_i * edge_mask_expanded).sum(
                    1
                ) / edge_mask_expanded.sum(1)

                if self.ablations["use_manifold_plucker"]:
                    loss_plucker = torch.nn.functional.mse_loss(
                        eps_plucker, eps_plucker_hat, reduction="none"
                    )
                    loss_plucker = (loss_plucker * edge_mask_expanded).sum(
                        1
                    ) / edge_mask_expanded.sum(1)

            else:
                loss_node_i = torch.nn.functional.mse_loss(
                    eps_node, eps_node_hat, reduction="none"
                ).mean(1)
                loss_edge_i = torch.nn.functional.mse_loss(
                    eps_edge, eps_edge_hat, reduction="none"
                ).mean(1)

                if self.ablations["use_manifold_plucker"]:
                    loss_plucker = torch.nn.functional.mse_loss(
                        eps_plucker, eps_plucker_hat, reduction="none"
                    ).mean(1)

        # Node losses
        node_loss_cardinal = 1
        loss_nodes = {}
        for feature, condition, index, increment in self.node_flags:
            if condition:
                node_loss_cardinal += 1
                weight = 1.0  # 10.0 if feature in {"r", "t"} else 1.0
                loss_nodes[feature] = weight * loss_node_i[:, index : index + increment]

        # Edge losses
        edge_loss_cardinal = 1
        loss_edges = {}
        for feature, condition, index, increment in self.edge_flags:
            if condition:
                # loss_cardinal += 1
                weight = 1.0  # 5.0 if feature in {"m", "p"} else 1.0
                loss_edges[feature] = weight * loss_edge_i[:, index : index + increment]

        # Combined losses
        if self.use_separate_loss:
            loss_node_i = (
                sum(loss_node.mean(-1) for loss_node in loss_nodes.values())
                / node_loss_cardinal
            )
            loss_edge_i = (
                sum(loss_edge.mean(-1) for loss_edge in loss_edges.values())
                / edge_loss_cardinal
            )
        else:
            # Warning: here equally weights all channels.
            loss_node_i = loss_node_i.mean(-1)
            loss_edge_i = loss_edge_i.mean(-1)

        # Add a plucker loss term in case the manifold parametrization is being used
        if self.ablations["use_manifold_plucker"]:
            loss_edges["p"] = loss_plucker.mean(-1)

        # Set output
        output["batch_loss"] = loss_node_i.mean() + loss_edge_i.mean()
        output["loss_v"] = loss_node_i.mean().detach()
        output["loss_e"] = loss_edge_i.mean().detach()
        output["loss_v_i"] = loss_node_i.detach()
        output["loss_e_i"] = loss_edge_i.detach()
        output["loss_v_oc"] = loss_nodes["oc"].mean().detach()
        output["loss_v_bb"] = loss_nodes["bb"].mean().detach()
        output["loss_v_t"] = loss_nodes["t"].mean().detach()
        output["loss_e_p"] = loss_edges["p"].mean().detach()
        output["loss_e_l"] = loss_edges["l"].mean().detach()

        if self.ablations["use_categorical_asset"]:
            output["loss_v_sa"] = loss_nodes["sa"].mean().detach()
        if self.ablations["use_categorical_body"]:
            output["loss_v_sb"] = loss_nodes["sb"].mean().detach()
        if self.ablations["use_categorical_joint"]:
            output["loss_e_j"] = loss_edges["j"].mean().detach()
        if self.ablations["use_node_orientation"]:
            output["loss_v_r"] = loss_nodes["r"].mean().detach()
        if self.ablations["use_manifold_plucker"]:
            output["loss_e_m"] = loss_edges["m"].mean().detach()
        if self.ablations["use_node_2D_preencoded_latent"]:
            output["loss_v_2d"] = loss_nodes["2d"].mean().detach()

        return output
