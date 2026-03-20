"""Shared fixtures for all tests."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.generate_fixtures import generate_sample_data, generate_schema


@pytest.fixture
def sample_schema():
    """Return a sample schema dict."""
    return generate_schema()


@pytest.fixture
def sample_dataframe():
    """Return a sample DataFrame with 30 rows."""
    return generate_sample_data(n=30, seed=42)


@pytest.fixture
def tmp_project(tmp_path, sample_schema, sample_dataframe):
    """Set up a temporary project directory with schema and dataset."""
    import forge.utils.config as cfg

    # Save original paths
    orig_project_root = cfg.PROJECT_ROOT
    orig_config = cfg.CONFIG_PATH
    orig_schema = cfg.SCHEMA_PATH
    orig_dataset = cfg.DATASET_PATH
    orig_reward = cfg.REWARD_MODEL_PATH
    orig_surrogate = cfg.SURROGATE_MODEL_PATH
    orig_proposals = cfg.PROPOSALS_DIR

    # Set up tmp paths
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    proposals_dir = data_dir / "proposals"
    proposals_dir.mkdir()

    # Write config
    config = {
        "ranks": {"labels": ["A", "B", "C"], "order": "descending"},
        "reward_model": {"algorithm": "mord_logistic_at", "regularization": 1.0},
        "surrogate_model": {"kernel": "matern", "nu": 2.5, "noise_variance": 0.01, "training_iterations": 50},
        "acquisition": {"function": "ucb", "kappa": 2.0},
        "optimizer": {"method": "differential_evolution", "max_iter": 100, "popsize": 10, "seed": 42},
    }
    with open(config_dir / "default.yaml", "w") as f:
        yaml.dump(config, f)

    # Write schema
    with open(data_dir / "schema.yaml", "w") as f:
        yaml.dump(sample_schema, f)

    # Write dataset
    sample_dataframe.to_csv(data_dir / "dataset.csv", index=False)

    # Monkey-patch paths
    cfg.PROJECT_ROOT = tmp_path
    cfg.CONFIG_PATH = config_dir / "default.yaml"
    cfg.SCHEMA_PATH = data_dir / "schema.yaml"
    cfg.DATASET_PATH = data_dir / "dataset.csv"
    cfg.REWARD_MODEL_PATH = models_dir / "reward_model.pkl"
    cfg.SURROGATE_MODEL_PATH = models_dir / "surrogate_model.pt"
    cfg.PROPOSALS_DIR = proposals_dir

    yield tmp_path

    # Restore original paths
    cfg.PROJECT_ROOT = orig_project_root
    cfg.CONFIG_PATH = orig_config
    cfg.SCHEMA_PATH = orig_schema
    cfg.DATASET_PATH = orig_dataset
    cfg.REWARD_MODEL_PATH = orig_reward
    cfg.SURROGATE_MODEL_PATH = orig_surrogate
    cfg.PROPOSALS_DIR = orig_proposals
