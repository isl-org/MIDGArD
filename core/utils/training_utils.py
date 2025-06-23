import yaml

# Attempt to import the faster C-based YAML loader
import json
import os
import rootutils
import subprocess


def load_config(config_file):
    """
    Load YAML configuration from the specified file.

    Args:
    - config_file (str): Path to the configuration file to be loaded.

    Returns:
    - dict: The configuration loaded from the YAML file.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Set the root path
    base_directory = rootutils.find_root(__file__, indicator=".project-root")
    rootutils.set_root(path=base_directory, pythonpath=True)
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    config["base_directory"] = str(base_directory)

    return config


def save_experiment_params(args, directory) -> None:
    """
    Save experiment parameters including current git commit hash to a JSON file.

    Args:
    - args (Namespace): Command-line arguments or any object with attributes.
    - directory (str): Directory where the parameters JSON file will be saved.
    """
    # Convert args to dictionary and ensure all values are strings
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    # Determine the current git directory and initial hash (fallback 'foo')
    git_dir = os.path.dirname(os.path.realpath(__file__))
    git_head_hash = "foo"
    try:
        # Try getting the current git commit hash
        git_head_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
    except subprocess.CalledProcessError:
        # Handle failure: move to the git directory and retry getting the commit hash
        cwd = os.getcwd()  # Current working directory to return later
        os.chdir(git_dir)
        git_head_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        os.chdir(cwd)  # Return to original directory

    # Update parameters with git commit hash
    params["git-commit"] = str(git_head_hash)

    # Replace empty string values with None for clarity in JSON output
    for k, v in list(params.items()):
        if v == "":
            params[k] = None

    # If there's a config file specified, load and merge its contents
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)

    # Write the parameters dictionary to a JSON file in the specified directory
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
