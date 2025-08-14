import h5py
import json
import logging
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch
import torchvision
from core.utils.data_utils import (
    load_json,
    load_normalized_grid_sdf_samples_from_mesh,
    create_bbox_3d,
)
from core.utils.image_processor import ImageProcessor, ImgProcCfg


class HybridDataset(torch.utils.data.Dataset):
    """
    A dataset container for graph, semantic, 2D and 3D data required to train the shape generator.
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
        self.config = config
        config_dict = config["dataset"]

        # Data paths
        self.base_directory = config["base_directory"]  # Defined in the training script
        self.data_directory = os.path.join(
            self.base_directory, config_dict["processed_data_directory"]
        )
        self.metadata_directory = os.path.join(
            self.base_directory, config_dict["metadata_directory"]
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

        self.mesh_ids = mesh_id[
            self.mode
            # mesh_id["train"] + mesh_id["val"] + mesh_id["test"]
        ]
        self.num_meshes = len(self.mesh_ids)

        # SDF or latent query data
        self.grid_resolution = config_dict.get("grid_resolution", 64)
        self.padding = config_dict.get("padding", 0.2)
        self.thresh_clamp_sdf = config_dict.get("thresh_clamp_sdf", 0.1)
        self.bb_factor = config["shape_generator"].get("bb_prior_factor", 0.02)

        # Condition flags
        diffusion_cfg = config["shape_generator"].get("denoising_diffusion", None)
        if diffusion_cfg is not None:
            self.text_condition = "txt" in diffusion_cfg.get("condition_type", "")
            self.image_condition = "img" in diffusion_cfg.get("condition_type", "")
            self.graph_condition = "graph" in diffusion_cfg.get("condition_type", "")
        else:
            self.text_condition = False
            self.image_condition = False
            self.graph_condition = False

        # Possibly load semantic data for text conditioning
        if self.text_condition:
            semantic_data_path = os.path.join(
                self.metadata_directory, config_dict.get("semantic_data", "")
            )
            data = load_json(semantic_data_path)
            self.semantic_data = data["semantic_data"]
        else:
            self.semantic_data = {}

        # Prepare image transforms if needed
        if self.image_condition:
            self.image_processor = ImageProcessor(ImgProcCfg())
        else:
            self.image_processor = None

        # Graph condition
        if self.graph_condition:
            pass

        # Build a dictionary of categories -> [IDs in that category]
        self.assets_ids = {}
        for cat, cat_splits in asset_id.items():
            if cat not in self.info["exp_cats"]:
                continue
            self.assets_ids[cat] = cat_splits[
                self.mode
            ]  # or combined train+val+test if desired

        # Create an inverted index: category_code -> category_name
        self.inverted_index = {}
        for cat, codes in self.assets_ids.items():
            for code in codes:
                if code not in self.inverted_index:
                    self.inverted_index[code] = cat

        # restrict mesh ids to the ones with existing paths and with included category
        new_mesh_ids = []
        for mesh_id in self.mesh_ids:
            cat_code = mesh_id.split("_")[0]
            instance_num = mesh_id.split("_")[1]
            # check if category is included
            if cat_code not in self.inverted_index:
                continue
            mesh_path = self._get_mesh_path(cat_code, instance_num, self.inverted_index[cat_code])
            if not os.path.exists(mesh_path):
                continue
            new_mesh_ids.append(mesh_id)
        self.mesh_ids = new_mesh_ids

        # Build final lists for latents, names, categories, and images
        self.name_list = []
        self.cats_list = []
        self.img_list = []

        self._gather_data_lists()

        logging.info(
            f"Loaded HybridDataset in '{self.mode}' mode with {len(self)} elements."
        )

        # Build class weights for WeightedRandomSampler (if needed)
        self.class_weight_list = self._build_class_weights()
        

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Total number of valid meshes in the dataset.
        """
        return len(self.mesh_ids)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the data and its metadata.
        """
        ret = {"index": index}
        mesh_id = self.name_list[index]
        category = self.cats_list[index]
        cat_code, instance_num = mesh_id.split("_")

        # Compute implicit SDF for the shape
        extent = self._load_sdf_data(cat_code, instance_num, category, ret)

        # Condition: text
        if self.text_condition:
            ret["txt"] = self._build_text_condition(mesh_id)

        # Condition: image
        if self.image_condition:
            img_list_for_mesh = self.img_list[index]
            if img_list_for_mesh:
                # Example: pick 1 random image
                random_ix = np.random.randint(len(img_list_for_mesh))
                im_path = img_list_for_mesh[random_ix]
                ret["img"] = self.image_processor(im_path, train=self.mode == "train")
                ret["img_path"] = im_path
            else:
                ret["img"] = None
                ret["img_path"] = ""

        # Condition: graph
        if self.graph_condition:
            # For demonstration, we here just store bounding box extent as "graph" condition
            ret["graph"] = extent

        # Metadata
        meta_info = {
            "meshid": mesh_id,
            "viz_id": f"{mesh_id}-{index}",
            "mode": self.mode,
            "category": category,
        }

        return ret, meta_info

    def _prepare_image_transforms(self):
        """
        Define image transformations and optional data augmentation for training vs. inference.
        Args:
            None
            
        Returns:
            None
        """
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(mean, std)
        self.resize = torchvision.transforms.Resize((256, 256), antialias=True)

        if self.mode == "train":
            self.transforms_color = torchvision.transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3
            )
            
            self.transforms_affine = torchvision.transforms.RandomAffine(
                degrees=0,
                scale=(0.7, 1.25),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            self.transforms_flip = torchvision.transforms.RandomHorizontalFlip()
        else:
            # For inference/validation
            self.transforms_color = None
            self.transforms_affine = None
            self.transforms_flip = None



    def _gather_data_lists(self):
        """
        Build lists of:
         - name_list:  mesh_id strings
         - cats_list:  category types
         - img_list:   image paths
        """
        lst_latent = {key: [] for key in self.assets_ids}
        lst_name = {key: [] for key in self.assets_ids}
        lst_img = {key: [] for key in self.assets_ids}
        for mesh_id in self.mesh_ids:

            category_code, instance_number = mesh_id.split("_")

            # Find which category in asset_id dict
            if category_code in self.inverted_index:
                asset_type = self.inverted_index[category_code]
                self.name_list.append(mesh_id)
                self.cats_list.append(asset_type)

                # If using images, gather all .png files in image directory
                if self.image_condition:
                    render_img_dir = os.path.join(
                        self.data_directory,
                        asset_type,
                        category_code,
                        "images",
                        instance_number,
                    )
                    if not os.path.exists(render_img_dir):
                        logging.debug(f"No image directory: {render_img_dir}")
                        self.img_list.append([])
                        continue
                    # render_img_list = [
                    #     os.path.join(render_img_dir, f)
                    #     for f in os.listdir(render_img_dir)
                    #     if f.endswith(".png")
                    # ]
                    render_img_list = [
                        os.path.join(
                            render_img_dir, "1.png"
                        )  # Following Nina's giudeline to use an image slightly on the side for better performance
                    ]
                    self.img_list.append(render_img_list)
                else:
                    self.img_list.append([])

        # Optionally, log category stats:
        cat_counts = {}
        for cat in self.info["exp_cats"]:
            cat_counts[cat] = sum(1 for c in self.cats_list if c == cat)
            logging.info(f"  [Category {cat}]: {cat_counts[cat]} samples")
        print(f"Total samples in {self.mode}", np.sum(list(cat_counts.values())), f"with {len(self.mesh_ids)} mesh ids")

    def _build_class_weights(self):
        """
        Build per-sample weights based on category frequency for WeightedRandomSampler usage.
        Returns a list of float weights, one per dataset sample.

        Args:
            None

        Returns:
            list: A list of weights, one per dataset sample.
        """
        from collections import Counter

        # Count occurrences of each category
        cat_counter = Counter(self.cats_list)
        # Inverse frequency weighting
        cat_weights = {cat: 1.0 / count for cat, count in cat_counter.items()}

        # Build a weight list aligned with self.cats_list
        return [cat_weights[cat] for cat in self.cats_list]

    def _get_mesh_path(self, cat_code, instance_num, category):
        """
        Build the .obj file path (or other geometry file) for a specific mesh.

        Args:
            cat_code (str): The category code.
            instance_num (str): The instance number.
            category (str): The category name.

        Returns:
            str: The full path to the mesh file.
        """
        mesh_path = os.path.join(
            self.data_directory,
            category,
            cat_code,
            "manifold_meshes",
            f"{instance_num}.obj",
        )
        return mesh_path

    def _load_sdf_data(self, cat_code, instance_num, category, ret):
        """
        Compute or load the SDF data for the mesh at the given index.

        Args:
            cat_code (str): The category code.
            instance_num (str): The instance number.
            category (str): The category name.
            ret (dict): The return dictionary to store the SDF data.

        Returns:
            np.ndarray: The extent of the Oriented Bounding Box (OBB) of the mesh.
        """
        mesh_path = self._get_mesh_path(cat_code, instance_num, category)

        meshid = f"{cat_code}_{instance_num}"

        sdf_preprocessed_path = self.config["dataset"].get("sdf_preprocessed_path", None)
        if sdf_preprocessed_path is not None:
            sdf_path = os.path.join(sdf_preprocessed_path, f"sdf_{meshid}.h5")
        else:
            sdf_path = None
        
        if sdf_path is not None and os.path.exists(sdf_path):
            # print(meshid, "Loading sdf from", sdf_path)
            with h5py.File(sdf_path, "r") as hf:
                # load precomputed sdf
                self.res = self.grid_resolution
                if "SDF_oriented" in sdf_path:
                    sdf = torch.from_numpy(hf["pc_sdf_sample"][:]).reshape(1, 64, 64, 64)
                else:
                    sdf = torch.from_numpy(hf["pc_sdf_sample"][:])
                inside = sdf.squeeze() <= 0
                if not torch.any(inside):
                    extent = torch.tensor([1.0,1.0,1.0])
                    # need to set coords to 0 to have bb everywhere (because of +1)
                    min_coords, max_coords = torch.tensor([0,0,0]), torch.tensor([self.res-1,self.res-1,self.res-1])
                else:
                    coords = torch.nonzero(inside, as_tuple=False)
                    min_coords = torch.min(coords, dim=0).values
                    max_coords = torch.max(coords, dim=0).values
                    extent = (max_coords.to(dtype=torch.float32) - min_coords.to(dtype=torch.float32))/self.res
        else:
            if not os.path.exists(mesh_path):
                logging.warning(f"Missing mesh: {mesh_path}")
                ret["sdf"] = None
                return torch.tensor([1.0,1.0,1.0])

            sdf, extent, min_coords, max_coords, _ = (
                load_normalized_grid_sdf_samples_from_mesh(
                    file_path=mesh_path,
                    grid_resolution=self.grid_resolution,
                    thresh_clamp_sdf=self.thresh_clamp_sdf,
                    pad=self.padding
                )
            )

        ret["sdf"] = sdf
        ret["mesh_gt_path"] = mesh_path
        ret["bounding_box_sdf"] = (
            create_bbox_3d(
                min_coords,
                max_coords,
                val=0.5 * self.bb_factor,
                res=self.grid_resolution,
                as_bool=False,
            )
            .float()
            .unsqueeze(0)
        )
        return extent

    def _center_and_scale_mesh(self, mesh_o3d):
        """
        Center and scale the mesh in-place.

        Args:
            mesh_o3d (open3d.geometry.TriangleMesh): The mesh to process.

        Returns:
            None
        """
        center = mesh_o3d.get_center()
        mesh_o3d.translate(-center, relative=True)
        vertices = np.asarray(mesh_o3d.vertices)
        min_bound, max_bound = vertices.min(axis=0), vertices.max(axis=0)
        scale_factor = 2.0 / np.linalg.norm(max_bound - min_bound)
        mesh_o3d.scale(scale_factor, center=(0, 0, 0))

    def _build_text_condition(self, mesh_id: str) -> str:
        """
        Build a textual conditioning string if text_condition is enabled.
        Example format: "A drawer as component of a cabinet."

        Args:
            mesh_id (str): The mesh ID string.

        Returns:
            str: The textual conditioning string.
        """
        # self.semantic_data[mesh_id] might be [ <body_type>, <part_type>, <category> ] or similar
        # Adjust as needed for your semantics
        if mesh_id not in self.semantic_data:
            return ""
        body_type = self.semantic_data[mesh_id][1].lower()
        category = self.semantic_data[mesh_id][2].lower()
        return f"A {body_type} as part of a {category}."

    def get_sampler(self, shuffle=True, distributed=False):
        """
        Retrieve a sampler for balancing the dataset in training mode, similarly to the
        improved versions in other datasets.

        Args:
            shuffle (bool): Whether to shuffle the data.
            distributed (bool): Whether training is distributed.

        Returns:
            WeightedRandomSampler or None.
        """
        if self.mode != "train":
            return None

        if distributed:
            # If you still want to do distributed training with weighting, you'll need
            # a custom approach. For demonstration, we just use the stock DistributedSampler.
            from torch.utils.data.distributed import DistributedSampler

            logging.info("Returning a distributed sampler (no weighting).")
            return DistributedSampler(self, shuffle=shuffle)

        # Use WeightedRandomSampler for category balancing
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.class_weight_list,
            num_samples=len(self.class_weight_list),
            replacement=True,
        )
        logging.info("Using WeightedRandomSampler with inverse-frequency weighting.")
        return sampler
