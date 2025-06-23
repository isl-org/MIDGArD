#
# Copyright(c) 2025 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils/common.sh"

# A simple bash script to train the midgard shape_generator model
root_dir=$(find_root) || { log ERROR ".project-root not found"; exit 1; }
script_dir="$root_dir/scripts/shape_generator"
config_dir="$root_dir/config"
output_dir="$root_dir/output"
cd $script_dir

python train.py "$config_dir/shape_generator.yaml"