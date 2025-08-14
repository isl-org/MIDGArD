import argparse
import json
import numpy as np
import os
import sys
from tqdm import tqdm

from core.utils.data_utils import write_json


def process_folder(folder_path):
    """
    Process a single folder to extract semantic data from 'semantics.txt' and metadata from 'meta.json'.

    The function reads the file 'meta.json' to extract the model category and then reads
    'semantics.txt' to extract joint type and body type for each link. The processed data is stored
    in a dictionary where each key corresponds to a unique body name (derived from the folder name and link name)
    and the value is a list containing the joint type, lowercased body type, and model category.
    Additionally, sets of unique model categories and body types are updated.

    Args:
        folder_path (str): The path to the folder containing 'semantics.txt' and 'meta.json'.

    Returns:
        tuple:
            - data (dict): Dictionary mapping body names to a list [joint_type, body_type (lowercase), model_cat].
            - model_cats (set): Set of unique model categories encountered.
            - body_types (set): Set of unique body types encountered.
    """
    # Paths to the required files
    semantics_path = os.path.join(folder_path, "semantics.txt")
    meta_path = os.path.join(folder_path, "meta.json")

    data = {}
    model_cats = set()
    body_types = set()

    # Check if both files exist
    if os.path.exists(semantics_path) and os.path.exists(meta_path):
        # Read and parse 'meta.json'
        with open(meta_path, "r") as file:
            meta_data = json.load(file)
            anno_id = os.path.basename(folder_path)
            model_cat = meta_data.get("model_cat", "").lower()
            model_cats.add(model_cat)

        # Read and process 'semantics.txt'
        with open(semantics_path, "r") as file:
            for line in file:
                link_name, joint_type, body_type = line.strip().split()
                body_types.add(body_type.lower())
                body_name = anno_id + "_" + link_name.split("_")[1]
                data[body_name] = [joint_type, body_type.lower(), model_cat]
    else:
        print(f"Something went south in {folder_path}")

    return data, model_cats, body_types


def process_all_folders(root_directory):
    """
    Process all subfolders in the given root directory to extract semantic labels.

    This function iterates over every subfolder within the root directory, calls process_folder on each
    subfolder to extract semantic information, and aggregates the results into a single dataset.
    It returns the total number of processed folders along with consolidated data, unique model categories,
    and unique body types.

    Args:
        root_directory (str): The path to the root directory containing multiple subfolders.

    Returns:
        tuple:
            - N_folders (int): Total number of folders processed.
            - all_data (dict): Consolidated dictionary containing semantic data from all folders.
            - all_model_cats (set): Set of all unique model categories encountered.
            - all_body_types (set): Set of all unique body types encountered.
    """
    all_data = {}
    all_model_cats = set()
    all_body_types = set()
    N_folders = 0

    # Traverse through each subfolder in the root directory
    for folder_name in tqdm(os.listdir(root_directory)):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path):
            data, model_cats, body_types = process_folder(folder_path)
            all_data.update(data)
            all_model_cats.update(model_cats)
            all_body_types.update(body_types)
            N_folders += 1

    return N_folders, all_data, all_model_cats, all_body_types


def main(argv):
    """
    Main function for processing semantic data from a dataset of PartNet Mobility raw data.

    This function parses command-line arguments to determine the input data directory (containing the raw data)
    and the output directory for processed data. It then calls process_all_folders to extract semantic labels
    from each subfolder, prints summary statistics, and finally saves the consolidated semantic data, unique model
    categories, and unique body types to a json file.

    Args:
        argv (list of str): List of command-line arguments.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Parse the datafiles to build a dataset of semantic labels."
    )
    parser.add_argument(
        "data_directory",
        type=str,
        help="Path to the folder containing the Partnet Mobility raw data.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="Path to the folder containing the processed data.",
    )
    args = parser.parse_args(argv)

    # Extract semantic data
    (
        N_folders,
        semantic_data,
        unique_model_cats,
        unique_body_types,
    ) = process_all_folders(args.data_directory)

    # Get the counts
    model_cat_count = len(unique_model_cats)
    body_type_count = len(unique_body_types)
    print("-------------------------------------------------")
    print(
        f"Processed {N_folders} folders for a total of {model_cat_count} distinct model categories and {body_type_count} unique body types."
    )
    print(f"Model categories: {unique_model_cats}")
    print(f"Body types: {unique_body_types}")
    print("-------------------------------------------------")

    # Convert sets to lists for JSON serialization
    results = {
        "semantic_data": semantic_data,
        "model_cats": list(unique_model_cats),
        "body_types": list(unique_body_types),
    }

    # Save the consolidated data to a .json file
    write_json(results, os.path.join(args.output_directory, "semantic_data.json"))


if __name__ == "__main__":
    main(sys.argv[1:])
