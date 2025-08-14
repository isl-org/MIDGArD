#
# Copyright(c) 2025 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils/common.sh"

# Activate midgard conda environment
activate_env

# Find the project root
root_dir=$(find_root) || { log ERROR ".project-root not found"; exit 1; }

# Define directories based on the determined root directory.
dataset_dir="$root_dir/dataset/PartNetMobility"
script_dir="$root_dir/scripts/dataset"

# Path to the Manifold executable
manifold_exec_path="$root_dir/core/utils/Manifold/build/manifold"

# Change directory to the script directory where the Python script is located.
cd "${script_dir}" 

case ${1:-all} in
  graph)     build_graph_dataset   "$dataset_dir" 10000 10000 "$manifold_exec_path" ;;
  image)     build_image_dataset   "$dataset_dir" 24    137   137 ;;
  manifold)  build_manifold_mesh_dataset "$dataset_dir" "$manifold_lib" 10000 ;;
  all)       build_graph_dataset   "$dataset_dir" 10000 10000 "$manifold_exec_path"
             build_image_dataset   "$dataset_dir" 24    137   137 ;;
  *)         echo "Usage: $0 [graph|image|manifold|all]" ; exit 2 ;;
esac
