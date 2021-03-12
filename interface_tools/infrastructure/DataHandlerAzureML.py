import logging
from typing import Dict, Generic, TypeVar

from azureml.core import Dataset, Workspace

T = TypeVar("T")

logger = logging.getLogger()


class DataHandlerAzureML(Generic[T]):
    def __init__(self) -> None:
        self.ws = None

    def load(self, config: Dict) -> T:
        self._get_workspace()
        dataset = Dataset.get_by_name(self.ws, name=config["name"])
        logger.info(f"Loaded dataset from {config['name']}")
        return dataset

    # We don't want to connect to the Azure ML workspace earlier than necessary.
    # This function will check whether you are connect to the workspace when
    # load/save is called, rather than, say, when the class object is created.
    def _get_workspace(self) -> None:
        if self.ws is None:
            self.ws = Workspace.from_config()
