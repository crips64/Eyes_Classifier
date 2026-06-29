"""Static validation of the local MinIO DVC contract."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dvc_contract():
    assert (ROOT / ".dvc" / "config").is_file()
    assert (ROOT / "data" / "reference.dvc").is_file()
    assert (ROOT / "eye_cnn_best_val_final.pth.dvc").is_file()
    assert (ROOT / "dvc.yaml").is_file()
    assert (ROOT / "dvc.lock").is_file()


def test_dvc_config_contains_no_credentials():
    content = (ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    assert "s3://mlops-eyes/dvc" in content
    assert "access_key_id" not in content
    assert "secret_access_key" not in content


def test_root_dvc_ignores_the_cookiecutter_template_pipeline():
    content = (ROOT / ".dvcignore").read_text(encoding="utf-8")
    assert "cookiecutter-mlops-eyes/" in content


def test_data_pointer_is_nonempty():
    content = (ROOT / "data" / "reference.dvc").read_text(encoding="utf-8")
    assert "nfiles: 3805" in content


def test_training_stage_tracks_the_config_it_reads():
    content = (ROOT / "dvc.yaml").read_text(encoding="utf-8")
    assert "python -m backend.src.train --config configs/train.yaml" in content
    assert "configs/train.yaml:" in content
    assert "--fast-dev-run" not in content
    assert "always_changed: true" in content
    assert "      - data/incoming\n" not in content


def test_ci_and_manifests_do_not_pin_a_student_account():
    workflow = (ROOT / ".github" / "workflows" / "ci-cd.yml").read_text(
        encoding="utf-8"
    )
    manifests = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "k8s").rglob("*.yaml")
    )
    assert "artembotsman" not in workflow.lower()
    assert "artembotsman" not in manifests.lower()
    assert "GITHUB_REPOSITORY_OWNER" in workflow
