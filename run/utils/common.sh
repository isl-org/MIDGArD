#!/bin/bash
set -e
IFS=$'\n\t'

################################################################################
#  Logging                                                                     #
################################################################################
log() {
    local type=$1
    local message=$2
    local len=${#message}
    local border=$(printf '%*s' "$((len + 4))" | tr ' ' '#')

    # Colors
    local NC="\033[0m" # No Color
    local GREEN="\033[0;32m"
    local YELLOW="\033[0;33m"
    local RED="\033[0;31m"

    # Choosing color based on message type
    local COLOR="$NC"
    case "$type" in
        DEBUG) COLOR="$NC" ;;
        INFO) COLOR="$GREEN" ;;
        WARNING) COLOR="$YELLOW" ;;
        ERROR) COLOR="$RED" ;;
    esac

    # Printing the message
    echo -e "${COLOR}${border}${NC}"
    echo -e "${COLOR}# $message #${NC}"
    echo -e "${COLOR}${border}${NC}"
}

################################################################################
#  Project root detection                                                      #
################################################################################
find_root() {
    local dir=${1:-"$PWD"}
    while [[ $dir != / ]]; do
        [[ -e "$dir/.project-root" ]] && { printf '%s\n' "$dir"; return; }
        dir=$(dirname "$dir")
    done
    return 1
}

################################################################################
#  Conda helpers                                                               #
################################################################################
readonly PYTHON_VERSION="3.12.8"
readonly ENV_NAME=midgard
detect_conda_prefix() {
    for d in "$HOME"/{mambaforge,miniforge3,anaconda3}; do
        [[ -d $d ]] && { echo "$d"; return; }
    done
    log ERROR "No valid Miniforge or Anaconda installation found." && exit 1
}

activate_env() {
    local prefix; prefix=$(detect_conda_prefix)
    log INFO  "Activating $ENV_NAME from $prefix"
    # shellcheck disable=SC1090
    source "$prefix/bin/activate" "$ENV_NAME"
}

################################################################################
#  Re-usable dataset tasks                                                     #
################################################################################
build_graph_dataset() {
    local dataset_dir=$1 point_samples=$2 tri=$3 manifold=$4
    log INFO "Generating graph dataset"
    python build_graph_dataset.py              \
        "$dataset_dir"                         \
        --manifold_exec_path "$manifold"       \
        --point_samples "$point_samples"       \
        --target_number_of_triangles "$tri"
    log INFO "Graph dataset generated"
}

build_image_dataset() {
    local dataset_dir=$1 views=$2 H=$3 W=$4
    log INFO "Generating image dataset"
    python build_image_dataset.py "$dataset_dir" \
        --viewpoints_num "$views" --height "$H" --width "$W"
    log INFO "Image dataset generated"
}

build_manifold_mesh_dataset() {
    local dataset_dir=$1 manifold_dir=$2 tri=$3
    log INFO "Generating manifold-mesh dataset"
    python build_manifold_mesh_dataset.py "$dataset_dir" \
        --manifold_lib "$manifold_dir"                   \
        --target_number_of_triangles "$tri"
    log INFO "Manifold-mesh dataset generated"
}
