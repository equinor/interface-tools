import logging
import os
from typing import Any, Dict, Generic, TypeVar

import joblib
import pandas as pd
from azureml.core import Dataset, Workspace
from azureml.core.model import Model

T = TypeVar("T")

logger = logging.getLogger()


class DataHandlerAzureML(Generic[T]):
    def __init__(self) -> None:
        self.ws = None

    def save(self, content: T, config: Dict) -> None:
        self._get_workspace()
        if config["file_type"] == "pickle":
            return self._save_pickle(content, config["name"])
        else:
            raise ValueError(f'File type of value {config["file_type"]} not supported')

    def load(self, config: Dict) -> T:
        self._get_workspace()
        if config["file_type"] == "dataframe":
            return self._load_dataframe(config["name"])
        else:
            raise ValueError(f'File type of value {config["file_type"]} not supported')

    def _save_pickle(self, content: Any, identifier: str) -> None:
        filename = f"{identifier}.pkl"
        joblib.dump(content, filename)
        Model.register(self.ws, filename, identifier)
        logger.info(f"Saved pickle file from {identifier}")
        os.remove(filename)

    def _load_dataframe(self, identifier: str) -> pd.DataFrame:
        dataset = Dataset.get_by_name(self.ws, name=identifier)
        df = dataset.to_pandas_dataframe()
        logger.info(f"Loaded {len(df)} rows DataFrame from {identifier}")
        return df

    # We don't want to connect to the Azure ML workspace earlier than necessary.
    # This function will check whether you are connect to the workspace when
    # load/save is called, rather than, say, when the class object is created.
    def _get_workspace(self) -> None:
        if self.ws is None:
            self.ws = Workspace.from_config()
