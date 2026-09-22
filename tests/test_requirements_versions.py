from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILES = [
    ROOT / "requirements_apple.txt",
    ROOT / "requirements_intel.txt",
    ROOT / "requirements_nvidia.txt",
]


def test_platform_requirements_pin_requested_versions() -> None:
    """Ensure the hardware-specific install requirements stay in sync."""
    for requirements_file in REQUIREMENTS_FILES:
        content = requirements_file.read_text()
        assert "torch==2.10.0" in content
        assert "hydra-core==1.3.4" in content
        assert "torch-2.10.0+" in content
