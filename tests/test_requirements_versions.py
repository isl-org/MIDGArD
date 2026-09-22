from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PINS = {
    "requirements_apple.txt": {
        "torch==": "torch==2.10.0",
        "torchvision==": "torchvision==0.25.0",
        "torchaudio==": "torchaudio==2.10.0",
        "hydra-core==": "hydra-core==1.3.4",
        "-f ": "-f https://data.pyg.org/whl/torch-2.10.0+cpu.html",
    },
    "requirements_intel.txt": {
        "torch==": "torch==2.10.0",
        "torchvision==": "torchvision==0.25.0",
        "torchaudio==": "torchaudio==2.10.0",
        "hydra-core==": "hydra-core==1.3.4",
        "-f ": "-f https://data.pyg.org/whl/torch-2.10.0+cpu.html",
    },
    "requirements_nvidia.txt": {
        "torch==": "torch==2.10.0",
        "torchvision==": "torchvision==0.25.0",
        "torchaudio==": "torchaudio==2.10.0",
        "hydra-core==": "hydra-core==1.3.4",
        "-f ": "-f https://data.pyg.org/whl/torch-2.10.0+cu128.html",
    },
}


def test_platform_requirements_pin_requested_versions() -> None:
    """Ensure the hardware-specific install requirements stay in sync."""
    for filename, expected_lines in EXPECTED_PINS.items():
        requirements_file = ROOT / filename
        lines = [line.strip() for line in requirements_file.read_text().splitlines()]

        for prefix, expected_line in expected_lines.items():
            matching_lines = [line for line in lines if line.startswith(prefix)]
            assert matching_lines == [expected_line]
