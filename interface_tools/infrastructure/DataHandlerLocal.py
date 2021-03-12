import logging
import os
from pathlib import Path
from typing import Callable, Dict, Generic, TypeVar

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
