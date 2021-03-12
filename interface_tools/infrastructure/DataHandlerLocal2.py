import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Generic, TypeVar

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from interface_tools.infrastructure.data_definitions import FileType

logger = logging.getLogger()


T = TypeVar("T")


class DataHandlerLocal(Generic[T]):
    def save(
        self, config: Dict, base_path: Path, save_lambda: Callable[[Path], None]
    ) -> None:
        if "relative_path" in config:
            path = base_path / config["relative_path"]
        else:
            path = base_path
        if not os.path.isdir(path):
            os.mkdir(path)
            logger.info(f"Created working dir {path}")

        save_lambda(path / config["name"])

    def load(
        self, config: Dict, base_path: Path, load_lambda: Callable[[Path], T]
    ) -> T:
        if "relative_path" in config:
            path = base_path / config["relative_path"]
        else:
            path = base_path
        return load_lambda(path / config["name"])
