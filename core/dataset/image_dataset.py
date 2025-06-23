from collections import defaultdict
import json
import logging
import os
from PIL import Image
import torch
from core.utils.data_utils import load_json
from core.utils.image_processor import ImageProcessor, ImgProcCfg


class ImageDataset(torch.utils.data.Dataset):
    """
    A dataset container for image data required to train the object latent feature generator.
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

        self.mesh_ids = (
            # self.mode
            mesh_id["train"]
            + mesh_id["val"]
            + mesh_id["test"]
        )
        self.num_meshes = len(self.mesh_ids)

        # Build a dictionary of all categories -> [list of asset codes]
        self.assets_ids = {}
        for cat, cat_splits in asset_id.items():
            # Combine train/val/test for demonstration
            self.assets_ids[cat] = (
                cat_splits["train"] + cat_splits["val"] + cat_splits["test"]
            )

        # Create an inverted index: (category_code -> asset_type)
        self.inverted_index = {}
        for asset_type, codes in self.assets_ids.items():
            for code in codes:
                if code not in self.inverted_index:
                    self.inverted_index[code] = asset_type

        # Build transform pipeline
        self.image_processor = ImageProcessor(ImgProcCfg())

        # Build lists of image paths and categories
        self.name_list = []
        self.cats_list = []
        self.img_list = []

        self._gather_image_paths()

        logging.info(
            f"Loaded ImageDataset in '{self.mode}' mode with {len(self)} elements."
        )

        # Build balancing weights for WeightedRandomSampler if desired
        self.class_weight_list = self._build_class_weights()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Total number of valid images in the dataset.
        """
        return len(self.img_list)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the data and its metadata.
        """
        # Data dictionary
        ret = {"index": index}
        image_id = self.name_list[index]

        # There's exactly one path in self.img_list[index] based on your code
        im_path = self.img_list[index][0]

        # Apply transformations (padding, resizing, normalization, etc.)
        ret["img"] = self.image_processor(im_path, train=self.mode == "train")
        ret["img_path"] = im_path

        # Metadata
        meta_info = {
            "imageid": image_id,
            "viz_id": f"{image_id}-{index}",
            "mode": self.mode,
            "category": self.cats_list[index],
        }

        return ret, meta_info

    def _gather_image_paths(self):
        """
        Build lists of:
         - name_list:  mesh_id strings
         - cats_list:  category types
         - img_list:   image paths
        """
        lst_name = {key: [] for key in self.assets_ids}
        lst_img = {key: [] for key in self.assets_ids}
        for mesh_id in self.mesh_ids:

            category_code, instance_number = mesh_id.split("_")

            # If the mesh is in one of the considered categories
            if category_code in self.inverted_index:
                asset_type = self.inverted_index[category_code]

                render_img_dir = os.path.join(
                    self.data_directory,
                    asset_type,
                    str(category_code),
                    "images",
                    instance_number,
                )

                # Consider all the images
                # name_list = [
                #     mesh_id + "_" + f for f in os.listdir(render_img_dir) if ".png" in f
                # ]

                # Consider only the front view
                name_list = [
                    mesh_id
                    + "_1.png"  # Following Nina's giudeline to use an image slightly on the side for better performance
                ]
                lst_name[asset_type].append(name_list)

                # Consider all the images
                # render_img_list = [
                #     os.path.join(render_img_dir, f)
                #     for f in os.listdir(render_img_dir)
                #     if ".png" in f
                # ]

                # Consider only the front view
                render_img_list = [os.path.join(render_img_dir, "1.png")]
                lst_img[asset_type].append(render_img_list)

        for c in self.info["exp_cats"]:
            self.name_list += lst_name[c]
            self.cats_list += [c] * len(lst_img[c])
            self.img_list += lst_img[c]
            print("[*] %d samples for %s." % (len(lst_img[c]), c))

    def _build_class_weights(self):
        """
        Build per-sample weights for WeightedRandomSampler
        based on the frequency of each category in self.cats_list.

        Args:
            None

        Returns:
            list: A list of weights for each sample in the dataset.
        """
        # 1) Count the number of samples per category
        cat_counts = defaultdict(int)
        for cat in self.cats_list:
            cat_counts[cat] += 1

        # 2) Inverse-frequency weighting: weight_cat = 1.0 / count_cat
        cat_weights = {}
        for cat, cnt in cat_counts.items():
            cat_weights[cat] = 1.0 / cnt

        # 3) Build a weight list for each sample in the dataset
        class_weight_list = [cat_weights[cat] for cat in self.cats_list]
        return class_weight_list

    def get_sampler(self, shuffle=True, distributed=False):
        """
        Returns a sampler object for the dataset, following the style of the GraphDataset.
        If 'train' mode, it returns a WeightedRandomSampler for balancing categories.
        Otherwise, returns None.

        Args:
            shuffle (bool): Whether to shuffle the data.
            distributed (bool): Whether training is distributed.
        """
        if self.mode != "train":
            return None

        if distributed:
            # For distributed, you'd typically return DistributedSampler
            # but you might want a custom approach if you want weighted sampling.
            from torch.utils.data.distributed import DistributedSampler

            logging.info("Returning a distributed sampler (no weighting).")
            return DistributedSampler(self, shuffle=shuffle)

        # If you want balancing across categories
        # WeightedRandomSampler picks samples proportionally to these weights
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.class_weight_list,
            num_samples=len(self.class_weight_list),
            replacement=True,
        )
        logging.warning(
            "Using WeightedRandomSampler with per-category inverse frequency."
        )
        return sampler
