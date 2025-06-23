#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Demo script: build Manifold and run it on Stanford's bunny mesh
# ---------------------------------------------------------------------------
# - Safe‑mode flags (`set -Eeuo pipefail`) so any failure aborts immediately.
# - Uses `$(nproc)` for auto‑parallel `make`.
# - Works no matter where it is invoked from (path‑agnostic).
# - Skips re‑download / rebuild when artefacts already exist.
# ---------------------------------------------------------------------------

set -Eeuo pipefail
IFS=$'\n\t'

# ---------- Paths ----------------------------------------------------------
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

WORK_DIR="$ROOT_DIR/test"
BUILD_DIR="$ROOT_DIR/build"

OBJ_URL="https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"
OBJ_FILE="$WORK_DIR/bunny.obj"
OUT_FILE="$WORK_DIR/manifold.obj"

MANIFOLD_BIN="$BUILD_DIR/manifold"

# ---------- Pre‑flight checks ---------------------------------------------
for cmd in wget cmake make; do
    command -v "$cmd" >/dev/null \
        || { echo "Error: '$cmd' is required but not installed." >&2; exit 1; }
done

# ---------- Download mesh (once) ------------------------------------------
mkdir -p "$WORK_DIR"
if [[ ! -f "$OBJ_FILE" ]]; then
    echo "Downloading Stanford bunny..."
    wget -q -O "$OBJ_FILE" "$OBJ_URL"
fi

# ---------- Configure & build Manifold ------------------------------------
if [[ ! -x "$MANIFOLD_BIN" ]]; then
    cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -- -j"$(nproc)"
fi

# ---------- Run Manifold ---------------------------------------------------
"$MANIFOLD_BIN" "$OBJ_FILE" "$OUT_FILE" 2000
echo "Manifold mesh written to: $OUT_FILE"

