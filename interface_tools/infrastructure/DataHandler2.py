from pathlib import Path
from typing import Callable, Dict, Generic, Optional, TypeVar

from numpy.lib.npyio import load

from interface_tools.infrastructure.DataHandlerAzureML2 import DataHandlerAzureML
from interface_tools.infrastructure.DataHandlerLocal2 import DataHandlerLocal

from .data_definitions import StorageType

T = TypeVar("T")


class DataHandler(Generic[T]):
    def __init__(self) -> None:
        self.dhl = DataHandlerLocal[T]()
        self.dhaml = DataHandlerAzureML[T]()

    def save(
        self, config: Dict, base_path: Path, save_lambda: Callable[[Path], None]
    ) -> None:
        if config["storage_type"] == StorageType.LOCAL:
            self.dhl.save(config, base_path, save_lambda)
        elif config["storage_type"] == StorageType.AZ_ML_DS:
            raise NotImplementedError("Saving files to Azure ML is not supported")
        elif config["storage_type"] == StorageType.AZ_ST_ACC:
            raise NotImplementedError(
                "Azure file share storage is not currently implemented (see AB#39504)"
            )
        else:
            raise ValueError(
                f'Storage type of value {config["storage_type"]} not supported'
            )

    def load(
        self,
        config: Dict,
        base_path: Optional[Path] = None,
        load_lambda: Optional[Callable[[Path], T]] = None,
    ) -> T:
        if config["storage_type"] == StorageType.LOCAL:
            if base_path is None or load_lambda is None:
                raise ValueError(
                    "Base path or load lambda cannot be None when loading local files"
                )
            return self.dhl.load(config, base_path, load_lambda)
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
