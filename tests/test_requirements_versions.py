from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REQUIREMENTS = ROOT / "requirements_nvidia.txt"
SHARED_PREFIXES = ("torch==", "torchvision==", "torchaudio==", "hydra-core==")
PYG_INDEX_SUFFIXES = {
    "requirements_apple.txt": "cpu",
    "requirements_intel.txt": "cpu",
    "requirements_nvidia.txt": "cu128",
}


def _read_lines(requirements_file: Path) -> list[str]:
    return [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
    ]


def _get_single_matching_line(lines: list[str], prefix: str) -> str:
    matching_lines = [line for line in lines if line.startswith(prefix)]
    assert len(matching_lines) == 1
    return matching_lines[0]


def test_platform_requirements_pin_requested_versions() -> None:
    """Ensure the hardware-specific install requirements stay in sync."""
    canonical_lines = _read_lines(CANONICAL_REQUIREMENTS)
    canonical_pins = {
        prefix: _get_single_matching_line(canonical_lines, prefix)
        for prefix in SHARED_PREFIXES
    }

    assert canonical_pins == {
        "torch==": "torch==2.10.0",
        "torchvision==": "torchvision==0.25.0",
        "torchaudio==": "torchaudio==2.10.0",
        "hydra-core==": "hydra-core==1.3.4",
    }

    torch_version = canonical_pins["torch=="].split("==", maxsplit=1)[1]

    for filename, pyg_suffix in PYG_INDEX_SUFFIXES.items():
        lines = _read_lines(ROOT / filename)
        for prefix, expected_line in canonical_pins.items():
            assert _get_single_matching_line(lines, prefix) == expected_line

        assert _get_single_matching_line(lines, "-f ") == (
            f"-f https://data.pyg.org/whl/torch-{torch_version}+{pyg_suffix}.html"
        )
