import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger()

_SOURCE_NAME = "src"
_BASE_CONFIG_FILE_NAME = "config.json"
# Set a limit on how many folder levels to search
_FOLDER_LIMIT = 4
_CONFIG_VAR_NAME = "CONFIG_FILE_PATH"

_DEPLOY_ENV = "PWD"
_DEPLOY_ENV_VALUE = "/var/azureml-app"
_DEPLOY_PATH = "./NGTT-price-model/src"

# Override automatic behaviour by changing these values
BASE_CONFIG: Optional[Dict] = None
CONFIG: Optional[Dict] = None
CONFIG_FILE_PATH: Optional[str] = None


def get_pipeline_config() -> Tuple[Dict, Dict, Path, Path]:
    """
    Automatically retrieves base config and pipeline config dictionaries

    Base config is the top level config file (normally named ``config.json``) that contains the relative path to the pipeline
    config file. The config file contains config regarding the pipeline and is normally located under the ``settings/`` folder.

    Set ``BASE_CONFIG = { your json config here }`` to override default behaviour and use a custom base config dictionary instead.
    Set ``CONFIG = { your json config here }`` to override default behaviour and use a custom config dictionary instead.
    Set ``CONFIG_FILE_PATH = relative/path/to/config.json`` to override default behaviour and use a custom config file instead.

    Setting ``CONFIG`` overrides ``CONFIG_FILE_PATH``.

    Examples:

    .. code-block:: python
        import config_utils

        # Override settings to use either one of these lines
        config_utils.BASE_CONFIG = { ... # dict with config }
        config_utils.CONFIG = { ... # dict with config }

        config_utils.CONFIG_FILE_PATH = "relative/path/to/config.json"

    :return: tuple of base config (dictionary) and root Path (where relative paths branch from)
    :rtype: Tuple[Dict, Dict]
    """
    root_path, config_path = find_project_paths()

    if BASE_CONFIG:
        base_config = BASE_CONFIG
    else:
        with open(config_path) as json_file:
            base_config = json.load(json_file)

    if CONFIG:
        config = CONFIG
    else:
        relative_path = base_config["pipeline_config_file"]
        if os.getenv(_CONFIG_VAR_NAME):
            relative_path = os.getenv(_CONFIG_VAR_NAME)
        elif CONFIG_FILE_PATH:
            relative_path = CONFIG_FILE_PATH
        with open(root_path / relative_path) as json_file:
            config = json.load(json_file)
        logger.info(f"Loaded the config from {root_path / relative_path}")

    return base_config, config, root_path, config_path


def find_project_paths() -> Tuple[Path, Path]:
    """Automatically searches higher level directories for the base config file and the root project directory

    :raises ValueError: when the base config file is not found
    :raises ValueError: when the root project directory is not found
    :return: tuple of base config (dictionary) and root Path (where relative paths branch from)
    :rtype: Tuple[Path, Path]
    """
    root_path: Optional[Path] = None
    config_path: Optional[Path] = None

    if os.getenv(_DEPLOY_ENV) == _DEPLOY_ENV_VALUE:
        os.chdir(_DEPLOY_PATH)

    working_dir = Path(os.getcwd())
    folder_limit = 0
    while not root_path:
        if working_dir.joinpath(_BASE_CONFIG_FILE_NAME).is_file():
            config_path = working_dir.joinpath(_BASE_CONFIG_FILE_NAME)

        if _SOURCE_NAME in os.listdir(working_dir):
            root_path = working_dir
            if not config_path:
                raise ValueError(f"Config file {_BASE_CONFIG_FILE_NAME} was not found")
            return root_path, config_path

        folder_limit += 1
        if folder_limit == _FOLDER_LIMIT:
            break
        working_dir = working_dir.parent
    raise ValueError("Root dir was not found")
