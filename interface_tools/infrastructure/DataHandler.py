from pathlib import Path
from typing import Dict, Generic, TypeVar

from interface_tools.infrastructure.DataHandlerAzureML import DataHandlerAzureML
from interface_tools.infrastructure.DataHandlerLocal import DataHandlerLocal

from .data_definitions import StorageType

T = TypeVar("T")


class DataHandler(Generic[T]):
    def __init__(self) -> None:
        self.dhl = DataHandlerLocal[T]()
        self.dhaml = DataHandlerAzureML[T]()

    def save(self, content: T, config: Dict, base_path: Path = None) -> None:
        if config["storage_type"] == StorageType.LOCAL:
            self.dhl.save(content, config, base_path)
        elif config["storage_type"] == StorageType.AZ_ML_DS:
            self.dhaml.save(content, config)
        elif config["storage_type"] == StorageType.AZ_ST_ACC:
            raise NotImplementedError(
                "Azure file share storage is not currently implemented (see AB#39504)"
            )
        else:
            raise ValueError(
                f'Storage type of value {config["storage_type"]} not supported'
            )

    def load(self, config: Dict, base_path: Path = None) -> T:
        if config["storage_type"] == StorageType.LOCAL:
            return self.dhl.load(config, base_path)
        elif config["storage_type"] == StorageType.AZ_ML_DS:
            return self.dhaml.load(config)
        elif config["storage_type"] == StorageType.AZ_ST_ACC:
            raise NotImplementedError(
                "Azure file share storage is not currently implemented (see AB#39504)"
            )
        else:
            raise ValueError(
                f'Storage type of value {config["storage_type"]} not supported'
            )
