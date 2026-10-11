from pathlib import Path

from pricing_reversal_harness.config import load_config


# tests/ → harness/ → openshift_study/ → config/
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_load_shipped_config_validates():
    cfg = load_config(CONFIG_DIR)
    assert cfg.models.models, "expected at least one model in models.yaml"
    assert cfg.datasets.datasets, "expected datasets configured"
    # every consolidated dataset must resolve
    dataset_ids = {d.id for d in cfg.datasets.datasets}
    for ds_id in cfg.experiment.consolidated_datasets:
        assert ds_id in dataset_ids, f"consolidated_datasets refers to {ds_id!r}"
    # every model with hosted-proxy pricing must be present
    hosted = cfg.pricing.hosted_proxy
    for m in cfg.models.models:
        assert m.id in hosted, f"model {m.id!r} missing hosted_proxy entry"
    assert cfg.experiment.repeated_trials >= 1
