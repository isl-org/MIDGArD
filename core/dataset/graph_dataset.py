import json
import logging
import numpy as np
import os
import torch
from tqdm import tqdm
from core.utils.data_utils import load_json, preprocess_and_pack


class GraphDataset(torch.utils.data.Dataset):
    """
    A dataset container for fully connected graphs representing articulated mechanisms.
    """

    def __init__(self, config: dict, mode: str) -> None:
        """
        Initializes the dataset object with configuration and mode.

        Args:
            config (dict): Configuration dictionary for dataset setup.
            mode (str): Mode of dataset usage, e.g., 'train' or 'test'.
        """
        super().__init__()
        self.mode = mode.lower()  # train, test...
        config_dict = config["dataset"]

        # Data paths
        self.base_directory = config["base_directory"]  # Defined in the training script
        self.data_directory = os.path.join(
            self.base_directory, config_dict["processed_data_directory"]
        )
        self.metadata_directory = os.path.join(
            self.base_directory, config_dict["metadata_directory"]
        )
        self.codebook_directory = os.path.join(
            self.base_directory, config_dict["codebook_directory"]
        )

        # Load metadata
        self.info = load_json(
            os.path.join(self.metadata_directory, config_dict["dataset_info_file"])
        )
        mesh_id = load_json(
            os.path.join(self.metadata_directory, config_dict["part_splits"])
        )
        asset_id = load_json(
            os.path.join(self.metadata_directory, config_dict["articulated_splits"])
        )

        # Ablation flags
        self.use_categorical_asset = config.get("use_categorical_asset", False)
        self.use_categorical_body = config.get("use_categorical_body", False)
        self.use_categorical_joint = config.get("use_categorical_joint", False)
        self.use_node_orientation = config.get("use_node_orientation", False)
        self.use_manifold_plucker = config.get("use_manifold_plucker", False)
        self.use_node_2D_preencoded_latent = config.get(
            "use_node_2D_preencoded_latent", False
        )

        # Maximum number of nodes in the graph
        self.num_nodes_max = config_dict["max_nodes"]

        # Indexes of the codebook embeddings
        self.embedding_index = mesh_id["train"] + mesh_id["val"] + mesh_id["test"]

        # Used in the generation process...
        self.training_embedding_index = mesh_id["train"]

        # Config the scale: if scale all, scale all V and E, otherwise scale only shapecode
        self.scale_all = config_dict.get("scale_all", False)

        # If use std, then use std, then use the maximum
        self.scale_mode = config_dict.get("scale_mode", "std")

        # Load the precomputed object latent code (object embeddings) stored in .npz format.
        self.semantic_data_path = os.path.join(
            self.metadata_directory, config_dict.get("semantic_data", None)
        )

        # Only training permute nodes
        self.permute_nodes = (
            config_dict.get("permute_nodes", False) and self.mode == "train"
        )

        # Load semantic data (body/asset categories)
        data = load_json(self.semantic_data_path)
        self.embedding = data["semantic_data"]
        self.embedding_model_cats = sorted(data["model_cats"])
        self.embedding_body_types = sorted(data["body_types"])

        # Load 2D latent codes if needed
        self.latent_2d_data_path = os.path.join(
            self.codebook_directory, config_dict.get("latent_2d_data", None)
        )
        if self.use_node_2D_preencoded_latent:
            data_2d = np.load(self.latent_2d_data_path, allow_pickle=True)
            self.latent_code_2d = data_2d["embedding"]  # [train, val, test]
            mesh_ids = data_2d["ids"]  # [train, val, test]
            N_train = len(mesh_id["train"])
            self.training_embedding_2d = self.latent_code_2d[:N_train]
            self.training_embedding_2d_ids = mesh_ids[:N_train]

            asset_types = [
                self.embedding[mesh_id][2] for mesh_id in self.training_embedding_2d_ids
            ]

            # Make sure the latent codes are in the authorzed categories ONLY
            self.training_embedding_2d = self.training_embedding_2d[
                np.isin(asset_types, self.info["exp_cats"])
            ]

            self.training_embedding_2d_ids = self.training_embedding_2d_ids[
                np.isin(asset_types, self.info["exp_cats"])
            ]

            if self.scale_mode == "std":
                self.embedding_scale_2d = data_2d["std"] + 1e-8
            elif self.scale_mode == "max":
                self.embedding_scale_2d = abs(self.latent_code_2d).max(axis=0) + 1e-8
            else:
                raise NotImplementedError(
                    f"scale_mode={self.scale_mode} not recognized"
                )

        # Collect graph paths and categories
        assets_ids = {}
        for cat, cat_ids in asset_id.items():
            assets_ids[cat] = cat_ids[self.mode]

        self.graph_path_list, self.name_list, self.cats_list = (
            self._build_graph_file_list(assets_ids)
        )

        # Cache data into memory
        self.data_list = []  # holds raw data in memory
        self.data_partnet_index_list = []
        self.num_nodes_list = []

        # For balancing:
        self.balance_cnt_dict = {k: [] for k in range(2, self.num_nodes_max + 1)}

        # Load & preprocess
        self._cache_data()

        # Weighted sampler
        self._build_balancing()

        logging.info(
            f"Loaded graph dataset in {mode} mode for a total of {len(self)} elements."
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Total number of valid graphs in the dataset.
        """
        return len(self.data_list)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the following dict:
                ret: dict with 'V', 'E', 'V_scale', 'E_scale', and other fields
                meta_info: dict with metadata such as 'partnet-m-id', 'mesh_name_list', etc.
        """
        node_dict, edge_dict = self.data_list[index]

        # 1) Preprocess & pack
        node_data, edge_data, node_map = preprocess_and_pack(
            node_dict,
            edge_dict,
            permute=self.permute_nodes,
            num_nodes_max=self.num_nodes_max,
            use_categorical_joint=self.use_categorical_joint,
            use_node_orientation=self.use_node_orientation,
            use_manifold_plucker=self.use_manifold_plucker,
        )

        # 2) Mask non-existent nodes
        node_data = node_data * node_data[:, 0:1]

        # 3) Scale features (if scale_all == True)
        if self.scale_all:
            # We do elementwise division using the scale
            node_data = node_data / self.node_features_scale.unsqueeze(0)
            edge_data = edge_data / self.edge_features_scale.unsqueeze(0)
            v_scale = self.node_features_scale
            e_scale = self.edge_features_scale
        else:
            # no scaling on geometric features; scale factor = 1
            v_scale = torch.ones_like(self.node_features_scale)
            e_scale = torch.ones_like(self.edge_features_scale)

        # 4) Categorical encoding for asset and body types
        if (self.use_categorical_asset or self.use_categorical_body) and (
            self.semantic_data_path is not None
        ):
            asset_label, part_label = self._encode_categorical_labels(
                index, node_data, node_map
            )
            # Combine them back into node_data
            # node_data: [mask(1), ...existing features...]
            v_init = node_data[:, :1]
            v_rest = node_data[:, 1:]
            scale_buffer = []
            components = [v_init]
            if self.use_categorical_asset:
                components.append(asset_label)
                scale_buffer += [1.0] * asset_label.shape[-1]
            if self.use_categorical_body:
                components.append(part_label)
                scale_buffer += [1.0] * part_label.shape[-1]
            components.append(v_rest)
            node_data = torch.cat(components, dim=-1)

            # Adjust v_scale to reflect the new dimension
            # The newly added categorical dims are unscaled
            v_scale = torch.cat(
                [v_scale[:1], torch.ones(len(scale_buffer)), v_scale[1:]]  # mask
            )

        # 5) If using a preencoded 2D latent, append it
        mesh_name_list = []
        if self.use_node_2D_preencoded_latent and (
            self.latent_2d_data_path is not None
        ):
            latents_2d = self._fetch_2d_latent(index, node_data, node_map, mesh_name_list)
            node_data = torch.cat([node_data, latents_2d], dim=-1)
            # Adjust v_scale to reflect new dims
            v_scale = torch.cat(
                [v_scale, torch.from_numpy(self.embedding_scale_2d).float()]
            )

        # 6) Final dictionary
        ret = {
            "V": node_data,
            "E": edge_data,
            "V_scale": v_scale,
            "E_scale": e_scale,
            "dataset_index": index,
        }

        # 7) Metadata
        meta_info = {
            "partnet-m-id": self.data_partnet_index_list[index],
            "joint_mapping": ["Revolute", "Prismatic", "Screw"],
            "mode": self.mode,
            # fill up to num_nodes_max if needed
            "mesh_name_list": mesh_name_list
            + [""] * (self.num_nodes_max - len(mesh_name_list)),
            "viz_id": f"partnet-{self.data_partnet_index_list[index]}-{index}",
        }

        return ret, meta_info

    def _build_graph_file_list(self, assets_ids: dict) -> tuple:
        """
        Build lists of graph paths, names, and category tags.

        Args:
            assets_ids (dict): Dictionary of asset IDs for each category.

        Returns:
            tuple: A tuple containing the following lists:
                graph_path_list: List of graph file paths.
                name_list: List of asset names.
                cats_list: List of asset categories.
        """
        graph_path_list = []
        name_list = []
        cats_list = []

        # For each category in the config, retrieve paths
        for cat in self.info["exp_cats"]:
            if cat not in assets_ids:
                continue
            cat_graph_paths = []
            cat_graph_names = []
            for code in assets_ids[cat]:
                graph_file = os.path.join(
                    self.data_directory, cat, str(code), "graph.npz"
                )
                cat_graph_paths.append(graph_file)
                cat_graph_names.append(code)

            graph_path_list += cat_graph_paths
            name_list += cat_graph_names
            cats_list += [cat] * len(cat_graph_paths)
            logging.info(f"Category {cat}: {len(cat_graph_paths)} samples.")

        return graph_path_list, name_list, cats_list

    def _cache_data(self):
        """
        Loads graph data, stores them in memory, and computes global scaling if needed.

        Args:
            None

        Returns:
            None
        """
        node_feats_all = []
        edge_feats_all = []

        # 1) Load & store graph data
        for graph_data_file, partnet_index, asset_cat in tqdm(
            zip(self.graph_path_list, self.name_list, self.cats_list),
            total=len(self.graph_path_list),
            desc="Caching graph data",
        ):
            data = np.load(graph_data_file, allow_pickle=True)
            node_features_dict = data["V"].tolist()
            edge_features_dict = data["E"].tolist()

            num_nodes = len(node_features_dict)
            assert num_nodes >= 2, "Graph must have at least 2 nodes."
            if num_nodes > self.num_nodes_max:
                logging.info(
                    f"Skipping {graph_data_file}: {num_nodes} nodes > max {self.num_nodes_max}."
                )
                continue

            # Store raw data
            self.data_list.append((node_features_dict, edge_features_dict))
            self.data_partnet_index_list.append(partnet_index)
            self.num_nodes_list.append(num_nodes)
            self.balance_cnt_dict[num_nodes].append(partnet_index)

            # Preprocess & pack (without permute) to compute scaling stats
            node_data, edge_data, _ = preprocess_and_pack(
                node_features_dict,
                edge_features_dict,
                permute=False,
                num_nodes_max=self.num_nodes_max,
                use_categorical_joint=self.use_categorical_joint,
                use_node_orientation=self.use_node_orientation,
                use_manifold_plucker=self.use_manifold_plucker,
            )
            node_feats_all.append(node_data)
            edge_feats_all.append(edge_data)

        # 2) Build node/edge arrays for scaling
        node_feats_all = np.concatenate(node_feats_all, axis=0)
        edge_feats_all = np.concatenate(edge_feats_all, axis=0)

        # 3) Compute global feature scales
        self._compute_scaling_stats(node_feats_all, edge_feats_all)

    def _compute_scaling_stats(self, node_feats_all, edge_feats_all):
        """
        Compute node & edge scaling stats (std or max).

        Args:
            node_feats_all (np.ndarray): All node features.
            edge_feats_all (np.ndarray): All edge features.

        Returns:
            None
        """
        node_existance_offset = 1  # first dim is existence
        if self.use_categorical_joint:
            edge_label_offset = 3  # chirality offset
        else:
            edge_label_offset = (
                3 + 5
            )  # chirality(3) + joint_label(5) if not categorical

        if self.scale_mode == "std":
            node_scale = node_feats_all[:, node_existance_offset:].std(axis=0) + 1e-8
            edge_scale = edge_feats_all[:, edge_label_offset:].std(axis=0) + 1e-8
        elif self.scale_mode == "max":
            node_scale = (
                np.abs(node_feats_all[:, node_existance_offset:]).max(axis=0) + 1e-8
            )
            edge_scale = (
                np.abs(edge_feats_all[:, edge_label_offset:]).max(axis=0) + 1e-8
            )
        else:
            raise NotImplementedError(f"Unknown scale_mode={self.scale_mode}")

        # existence offset features remain unscaled => prepend ones
        self.node_features_scale = np.concatenate(
            [np.ones(node_existance_offset), node_scale], axis=0
        )

        # chirality (+ label) offset remain unscaled => prepend ones
        self.edge_features_scale = np.concatenate(
            [np.ones(edge_label_offset), edge_scale], axis=0
        )

        # Convert them to torch tensors for faster repeated use
        self.node_features_scale = torch.from_numpy(self.node_features_scale).float()
        self.edge_features_scale = torch.from_numpy(self.edge_features_scale).float()

    def _build_balancing(self):
        """
        Compute self.class_weight and self.class_weight_list for WeightedRandomSampler.

        Args:
            None

        Returns:
            None
        """
        self.balance_cnt = {k: len(v) for k, v in self.balance_cnt_dict.items()}
        logging.info(f"Balance counts: {self.balance_cnt}")
        self.class_weight = [0.0, 0.0]  # placeholding indices for <2
        # weight = 1 / count for each node size
        for k in range(2, self.num_nodes_max + 1):
            if self.balance_cnt[k] > 0:
                self.class_weight.append(1.0 / self.balance_cnt[k])
            else:
                self.class_weight.append(0.0)

        self.class_weight_list = [self.class_weight[n] for n in self.num_nodes_list]

    def _encode_categorical_labels(self, index, node_data, node_map):
        """
        Builds one-hot encodings for asset and body type per node.

        Args:
            index (int): Index of the current graph.
            node_data (np.ndarray): Node data for the current graph.
            node_map (np.ndarray): Node mapping for the current graph.

        Returns:
            tuple: A tuple containing the following one-hot encoded labels:
                asset_label: One-hot encoded asset labels.
                part_label: One-hot encoded body type labels.
        """
        asset_vals = []
        part_vals = []
        for node_idx, occ in zip(node_map, node_data[:, 0]):
            if occ < 0.5:
                # non-existent node => dummy zero
                # or possibly last index => special "dummy"
                key = f"{self.data_partnet_index_list[index]}_{0}"
                asset_vals.append(
                    self.embedding_model_cats.index(self.embedding[key][2])
                )
                part_vals.append(len(self.embedding_body_types))
            else:
                key = f"{self.data_partnet_index_list[index]}_{node_idx}"
                asset_vals.append(
                    self.embedding_model_cats.index(self.embedding[key][2])
                )
                part_vals.append(
                    self.embedding_body_types.index(self.embedding[key][1])
                )

        # One-hot
        asset_vals = torch.tensor(asset_vals, dtype=torch.long)
        asset_label = torch.nn.functional.one_hot(
            asset_vals, num_classes=len(self.embedding_model_cats) + 1
        ).float()

        part_vals = torch.tensor(part_vals, dtype=torch.long)
        part_label = torch.nn.functional.one_hot(
            part_vals, num_classes=len(self.embedding_body_types) + 1
        ).float()

        # Zero out the dummy category
        asset_label[asset_label[:, -1] == 1] = 0
        asset_label = asset_label[:, :-1]  # drop the dummy column

        part_label[part_label[:, -1] == 1] = 0
        part_label = part_label[:, :-1]  # drop the dummy column

        return asset_label, part_label

    def _fetch_2d_latent(self, index, node_data, node_map, mesh_name_list):
        """
        Retrieves and scales the 2D latent code for each node in the current graph.

        Args:
            index (int): Index of the current graph.
            node_data (np.ndarray): Node data for the current graph.
            node_map (np.ndarray): Node mapping for the current graph.
            mesh_name_list (list): List of mesh names for the current graph.

        Returns:
            torch.Tensor: 2D latent code for each node in the current graph.
        """
        code_list = []
        for node_idx, occ in zip(node_map, node_data[:, 0]):
            if occ < 0.5:
                # non-existent node => fill with zeros
                code_list.append(np.zeros_like(self.latent_code_2d[0]))
            else:
                key = f"{self.data_partnet_index_list[index]}_{node_idx}"
                mesh_name_list.append(key)
                code_id = self.embedding_index.index(key)
                # scale by std (embedding_scale_2d)
                code_list.append(
                    self.latent_code_2d[code_id] / (self.embedding_scale_2d + 1e-8)
                )

        return torch.from_numpy(np.stack(code_list, axis=0)).float()

    def get_sampler(self):
        """
        Retrieve a weighted random sampler for balancing the dataset in training mode.

        Returns:
            WeightedRandomSampler or None: Returns the sampler object if conditions are met, or `None`
            if balancing is not required or if the mode is not set to "train".
        """
        if self.mode != "train":
            return None

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.class_weight_list,
            num_samples=len(self.class_weight_list),
            replacement=True,
        )
        logging.warning(
            "Using WeightedRandomSampler with per-category inverse frequency."
        )

        return sampler
