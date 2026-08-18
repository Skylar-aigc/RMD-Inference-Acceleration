from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_EN = (ROOT / "README.md").read_text(encoding="utf-8")
README_CN = (ROOT / "README_CN.md").read_text(encoding="utf-8")


def test_readme_referenced_local_files_exist():
    expected_paths = (
        "README_CN.md",
        "configs/train_wan_1_3b.yaml",
        "configs/train_wan_14b.yaml",
        "configs/train_wan_1_3b_7gpu.yaml",
        "scripts/train_7gpu.sh",
        "scripts/train_8gpu.sh",
        "examples/vidprom_prompts_2000.txt",
        "examples/VIDPROM_DATASET.md",
        "docs/training.md",
        "docs/inference.md",
        "docs/npu.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "assets/rmd_overview.png",
        "assets/wan_video_comparison.png",
    )

    for relative_path in expected_paths:
        assert relative_path in README_EN
        assert (ROOT / relative_path).exists(), relative_path

    assert "README.md" in README_CN


def test_readmes_document_runnable_entry_points_and_defaults():
    for readme in (README_EN, README_CN):
        for command in (
            "rmd-train",
            "rmd-infer",
            "rmd-demo",
            "accelerate launch",
            "pytest -q",
        ):
            assert command in readme

        assert "300" in readme
        assert "50" in readme
        assert "2,000" in readme
        assert "2603.06136" in readme


def test_training_uses_variadic_models_for_gradient_accumulation():
    source = (ROOT / "src/rmd/engines/train.py").read_text(encoding="utf-8")
    assert "accelerator.accumulate(*accumulated_models)" in source
