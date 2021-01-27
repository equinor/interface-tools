import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Generic, TypeVar

import joblib
import numpy as np
import pandas as pd

from interface_tools.infrastructure.data_definitions import FileType

logger = logging.getLogger()


T = TypeVar("T")


class DataHandlerLocal(Generic[T]):
    def save(self, content: T, config: Dict, base_path: Path = None) -> None:
        if "relative_path" in config:
            path = base_path / config["relative_path"]
        else:
            path = base_path
        if not os.path.isdir(path):
            os.mkdir(path)
            logger.info(f"Created working dir {path}")

        if config["file_type"] == FileType.DATAFRAME:
            self._save_dataframe(content, config["name"], path)
        elif config["file_type"] == FileType.HDF5:
            self._save_hdf5(content, config["name"], path)
        elif config["file_type"] == FileType.NUMPY_ARR:
            self._save_numpy_array(content, config["name"], path)
        elif config["file_type"] == FileType.PICKLE:
            self._save_pickle(content, config["name"], path)
        elif config["file_type"] == FileType.JSON:
            self._save_json(content, config["name"], path)
        elif config["file_type"] == FileType.HTML:
            self._save_html(content, config["name"], path)
        elif config["file_type"] == FileType.PNG:
            self._save_png(content, config["name"], path)
        elif config["file_type"] == FileType.MATPLOTLIB_PNG:
            self._save_matplotlib_png(content, config["name"], path)
        else:
            raise ValueError(f'File type of value {config["file_type"]} not supported')

    def load(self, config: Dict, base_path: Path = None) -> T:
        if "relative_path" in config:
            path = base_path / config["relative_path"]
        else:
            path = base_path
        if config["file_type"] == FileType.DATAFRAME:
            return self._load_dataframe(config["name"], path)
        elif config["file_type"] == FileType.HDF5:
            return self._load_hdf5(config["name"], path)
        elif config["file_type"] == FileType.NUMPY_ARR:
            return self._load_numpy_array(config["name"], path)
        elif config["file_type"] == FileType.PICKLE:
            return self._load_pickle(config["name"], path)
        elif config["file_type"] == FileType.JSON:
            return self._load_json(config["name"], path)
        else:
            raise ValueError(f'File type of value {config["file_type"]} not supported')

    def _save_numpy_array(
        self, data: np.array, identifier: str, working_dir: str
    ) -> None:
        outfile = Path(working_dir) / f"{identifier}.npy"
        np.save(outfile, f"{identifier}.npy")
        logger.info(f"Saved {len(data)} rows of numpy data to {outfile}")

    def _load_numpy_array(self, identifier: str, working_dir: str) -> np.array:
        infile = Path(working_dir) / f"{identifier}.npy"
        data = np.load(infile)
        logger.info(f"Loaded {len(data)} rows of numpy data from {infile}")
        return data

    def _save_pickle(self, model: Any, identifier: str, working_dir: str) -> None:
        outfile = Path(working_dir) / f"{identifier}.pkl"
        with open(outfile, "wb") as file:
            pickle.dump(model, file)
        logger.info(f"Saved model to {outfile}")

    def _load_pickle(self, identifier: str, working_dir: str) -> pd.DataFrame:
        infile = Path(working_dir) / f"{identifier}.pkl"
        with open(infile, "rb") as file:
            model = pickle.load(file)
        logger.info(f"Loaded model from {infile}")
        return model

    def _save_json(self, data: Any, identifier: str, working_dir: str) -> None:
        outfile = Path(working_dir) / f"{identifier}.json"
        with open(outfile, "w") as file:
            json.dump(data, file)
        logger.info(f"Saved JSON to {outfile}")

    def _load_json(self, identifier: str, working_dir: str) -> T:
        infile = Path(working_dir) / f"{identifier}.json"
        with open(infile, "rb") as file:
            data = json.load(file)
        logger.info(f"Loaded json from {infile}")
        return data

    def _save_dataframe(
        self, df: pd.DataFrame, identifier: str, working_dir: str
    ) -> None:
        outfile = Path(working_dir) / f"{identifier}.csv"
        df.to_csv(outfile)
        logger.info(f"Saved {len(df)} rows DataFrame to {outfile}")

    def _load_dataframe(self, identifier: str, working_dir: str) -> pd.DataFrame:
        infile = Path(working_dir) / f"{identifier}.csv"
        df = pd.read_csv(infile)
        logger.info(f"Loaded {len(df)} rows DataFrame from {infile}")
        return df

    def _save_hdf5(self, df: pd.DataFrame, identifier: str, working_dir: str) -> None:
        outfile = str(Path(working_dir) / f"{identifier}.hdf5")
        df.to_hdf(outfile, key="df", mode="w", format="fixed")
        logger.info(f"Saved {len(df)} rows DataFrame to {outfile}")

    def _load_hdf5(self, identifier: str, working_dir: str) -> pd.DataFrame:
        infile = Path(working_dir) / f"{identifier}.hdf5"
        data = pd.read_hdf(infile)
        logger.info(f"Loaded {len(data)} rows of raw data from {infile}")
        return data

    def _save_joblib(
        self, data: Any, identifier: str, working_dir: str, file_extension: str = "pkl"
    ) -> None:
        outfile = str(Path(working_dir) / f"{identifier}.{file_extension}")
        joblib.dump(value=data, filename=outfile)
        logger.info(f"Saved via joblib to f{outfile}")

    def _load_joblib(
        self, identifier: str, working_dir: str, file_extension: str = "pkl"
    ) -> Any:
        infile = str(Path(working_dir) / f"{identifier}.{file_extension}")
        data = joblib.load(filename=infile)
        logger.info(f"Loaded via joblib from f{infile}")
        return data

    def _save_html(self, data: Any, identifier: str, working_dir: str) -> None:
        outfile = str(Path(working_dir) / f"{identifier}.html")
        f = open(outfile, "w")
        f.write(data)
        f.close()

    def _save_png(self, data: Any, identifier: str, working_dir: str) -> None:
        outfile = str(Path(working_dir) / f"{identifier}.png")
        with open(outfile, "wb") as f:
            f.write(data)

    def _save_matplotlib_png(
        self, data: Any, identifier: str, working_dir: str
    ) -> None:
        """
        Matplotlib doesn't have a conventient way to convert to bytes, so as a workaround add this method which
        runs matplotlib's own savefig method on the data
        :param data: matplotlib figure
        :param identifier:
        :param working_dir:
        :return:
        """
        outfile = str(Path(working_dir) / f"{identifier}.png")
        data.savefig(outfile)
