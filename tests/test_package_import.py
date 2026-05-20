from active_matter_jepa.train import ViTJEPAConfig, _merge_hjepa_config, build_model


def test_analytics_modules_import():
    import importlib
    import sys
    from pathlib import Path

    analytics_dir = Path(__file__).resolve().parents[1] / "analytics"
    sys.path.insert(0, str(analytics_dir))
    try:
        for module_name in [
            "analyze_active_matter",
            "feature_baselines",
            "utils_io",
            "utils_plots",
            "utils_stats",
        ]:
            importlib.import_module(module_name)
    finally:
        sys.path.remove(str(analytics_dir))


def test_hjepa_default_config_has_three_levels():
    cfg = _merge_hjepa_config({})
    assert sorted(cfg["levels"]) == ["l1", "l2", "l3"]


def test_small_hjepa_model_builds():
    cfg = ViTJEPAConfig(
        model_type="h_jepa_multiscale",
        img_size=64,
        mlp_ratio=2,
        drop_path_rate=0.0,
        h_jepa={
            "levels": {
                "l1": {"patch_size": 8, "embed_dim": 32, "depth": 1, "num_heads": 4},
                "l2": {"patch_size": 16, "embed_dim": 32, "depth": 1, "num_heads": 4},
                "l3": {"patch_size": 32, "embed_dim": 32, "depth": 1, "num_heads": 4},
            },
            "predictor_depth": 1,
            "predictor_hidden_mult": 2,
            "frame_batch_size": 2,
        },
    )
    cfg.h_jepa = _merge_hjepa_config(cfg.h_jepa)
    model = build_model(cfg)
    assert model.parameter_report()["total_params"] > 0
