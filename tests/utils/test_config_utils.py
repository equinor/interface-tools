import os

from interface_tools.utils import config_utils


def test_get_pipeline_config_ShouldReturnValidPath_WhenRun() -> None:
    # act
    root_path, config_path = config_utils.find_project_paths()

    # assert
    assert os.path.isdir(root_path)
    assert os.path.isfile(config_path)


def test_get_pipeline_config_ShouldReturnValidConfig_WhenRun() -> None:
    # act
    base_config, config, _, _ = config_utils.get_pipeline_config()

    # assert
    assert base_config
    assert config
    assert type(base_config) == dict
    assert type(config) == dict


def test_get_pipeline_config_ShouldReturnValidConfig_WhenGivenCustomConfig() -> None:
    # arrange
    config_utils.CONFIG = {"test": 1}
    config_utils.BASE_CONFIG = {"test": 2}

    # act
    base_config, config, _, _ = config_utils.get_pipeline_config()

    # assert
    assert base_config["test"] == 2
    assert config["test"] == 1

    # Clean up
    config_utils.CONFIG = None
    config_utils.BASE_CONFIG = None


def test_get_pipeline_config_ShouldReturnSameConfig_WhenGivenSamePathAsConfig() -> None:
    # arrange
    base_config, config, _, _ = config_utils.get_pipeline_config()
    relative_path = base_config["pipeline_config_file"]

    # act
    config_utils.CONFIG_FILE_PATH = relative_path
    _, config_test, _, _ = config_utils.get_pipeline_config()

    # assert
    assert config == config_test

    # Clean up
    config_utils.CONFIG_FILE_PATH = None
