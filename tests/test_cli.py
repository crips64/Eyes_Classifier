"""Smoke tests for the backwards-compatible CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

from open_eyes_classifier import MediumEyeCNN

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "open_eyes_classifier.py"


def test_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--weights" in result.stdout


def test_predict_on_temp_grayscale_png(tmp_path: Path) -> None:
    image = tmp_path / "tiny.png"
    weights = tmp_path / "weights.pth"
    Image.new("L", (24, 24), color=128).save(image)
    torch.manual_seed(42)
    torch.save(MediumEyeCNN().state_dict(), weights)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(image),
            "--weights",
            str(weights),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert 0 <= float(result.stdout.strip().splitlines()[0]) <= 1
