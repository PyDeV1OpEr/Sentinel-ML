import pytest
import yaml
from pathlib import Path

from spectral.config.loader import get_default_config, load_config
from spectral.config.schema import PipelineConfig


class TestConfigLoader:
    def test_default_config_values(self):
        config = get_default_config()
        assert config.name == "default"
        assert config.window_size == 100
        assert len(config.detectors) == 3
        assert config.preprocessing.normalize is True

    def test_loads_valid_yaml(self, tmp_path):
        config_file = tmp_path / "test.yml"
        config_file.write_text(yaml.dump({
            "pipeline": {"name": "custom", "window_size": 200},
            "detectors": [
                {"name": "zscore", "params": {"threshold": 2.5}},
            ],
        }))

        config = load_config(config_file)
        assert config.name == "custom"
        assert config.window_size == 200
        assert config.detectors[0].params["threshold"] == 2.5

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yml")

    def test_empty_file_returns_defaults(self, tmp_path):
        config_file = tmp_path / "empty.yml"
        config_file.write_text("")

        config = load_config(config_file)
        assert config.name == "default"

    def test_partial_config_merges_with_defaults(self, tmp_path):
        config_file = tmp_path / "partial.yml"
        config_file.write_text(yaml.dump({
            "pipeline": {"name": "partial"},
        }))

        config = load_config(config_file)
        assert config.name == "partial"
        # should still have default preprocessing
        assert config.preprocessing.normalize is True
