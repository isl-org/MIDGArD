#
# Copyright(c) 2025 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/utils/common.sh"

# A simple bash script to evaluate the generative graph diffusion model
root_dir=$(find_root) || { log ERROR ".project-root not found"; exit 1; }
script_dir="$root_dir/scripts"
config_dir="$root_dir/config"
output_dir="$root_dir/output"
cd $script_dir

python midgard_evaluate.py "$config_dir/midgard_gen.yaml"