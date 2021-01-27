import logging
import tempfile
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from azureml._common.exceptions import AzureMLException
from azureml.core.run import Run
from matplotlib import pyplot as plt

logger = logging.getLogger()


class ExperimentLogger:
    def log_scalar_via_lookup(
        self, data: Dict, key: Union[str, Enum], prefix: Optional[str] = None
    ) -> None:
        raise NotImplementedError()

    @staticmethod
    def log_scalar(
        key: Union[str, Enum], value: Any, prefix: Optional[str] = None
    ) -> None:
        raise NotImplementedError()

    @staticmethod
    def log_df(df: pd.DataFrame, slug: str, description: str = None) -> None:
        raise NotImplementedError()

    @staticmethod
    def log_figure_matplotlib(figure: plt.Figure, slug: str) -> None:
        raise NotImplementedError()

    @staticmethod
    def log_figure_plotly(figure: go.Figure, slug: str) -> None:
        raise NotImplementedError()


class ExperimentLoggerAzureml(ExperimentLogger):
    @staticmethod
    def _log_scalar(key: Optional[str], value: Union[float, int, None]) -> None:
        run = Run.get_context()
        if key:
            # experimental, only log if value is non nan
            if pd.isnull(value):
                logger.debug(f"_log_scalar skipping logging Null value value:{value})")
            else:
                logger.debug(
                    f'_log_scalar DISABLED Logging via run.log("{key}", {value})'
                )
                run.log(key, value)

    @staticmethod
    def _prepare_key(key: Union[str, Enum]) -> Optional[str]:
        if isinstance(key, Enum):
            logger.debug(f"Converting enum to str {key} -> {key.value}")
            key_ = key.value
            assert type(key_) is str
        else:
            key_ = key
        return key_

    @staticmethod
    def _prepare_value(value: Any) -> Optional[Union[float, int]]:
        if type(value) is np.float64:
            value = float(value)
        if type(value) is float or type(value) is int:
            return value
        else:
            return None

    @staticmethod
    def _prepare_scalar(
        key: Union[str, Enum], value: Any, prefix: Optional[str] = None
    ) -> Tuple[Optional[str], Union[float, int, None]]:
        key_ = ExperimentLoggerAzureml._prepare_key(key)
        value_ = ExperimentLoggerAzureml._prepare_value(value)
        if value_ is None:
            key_ = None
        else:
            if prefix:
                key_ = f"{prefix}-{key_}"
        return key_, value_

    @staticmethod
    def log_scalar(
        key: Union[str, Enum], value: Any, prefix: Optional[str] = None
    ) -> None:
        """
        Log scalar value to AzureML experiment iff it can be cast to a compatible type
        :param key: key of scalar (can be enum or string)
        :param value: value (can be anything)
        :param prefix: (optional) identifier for scalar (e.g. "test set")
        :return: None
        """
        logger.debug(f"log_scalar key:{key} value:{value} prefix:{prefix} START")
        key_, value_ = ExperimentLoggerAzureml._prepare_scalar(key, value, prefix)
        ExperimentLoggerAzureml._log_scalar(key_, value_)
        logger.debug(f"log_scalar key:{key_} value:{value_} prefix:{prefix} END")

    @staticmethod
    def _prepare_scalar_via_lookup(
        data: Dict, key: Union[str, Enum], prefix: Optional[str] = None
    ) -> Tuple:
        value = data[key]
        return ExperimentLoggerAzureml._prepare_scalar(key, value)

    @staticmethod
    def log_scalar_via_lookup(
        data: Dict, key: Union[str, Enum], prefix: Optional[str] = None
    ) -> None:
        """
        Log scalar value to AzureML experiment iff the key and value can be cast to compatible types
        :param data: dictionary of key value pairs
        :param key: key for lookup
        :param prefix: (optional) identifier for scalar (e.g. "test set")
        :return: None
        """
        value = data[key]
        key_, value_ = ExperimentLoggerAzureml._prepare_scalar(key, value, prefix)
        ExperimentLoggerAzureml._log_scalar(key_, value_)

    @staticmethod
    def log_df(df: pd.DataFrame, slug: str, description: str = None) -> None:
        """
        :param df: dataframe of results
        :param slug: root name of metric (metric is logged as slug-COL where COL is the column name)
        :param description: string description of metric (e.g. "test")
        :return: None
        """
        run = Run.get_context()

        if type(df.index) == pd.DatetimeIndex:
            df.index = df.index.strftime("%Y-%m-%d %H:%M:%S %Z")
            if df.index.name is None:
                df.index.name = "timestamp"
            df = df.reset_index()

        # Log each column as a separate table to give maximum flexibility in visualization in AzureML experiment portal
        for column in df.columns:
            series = df[column]

            # experiment logging is fussy about formats, so log each column inside a try catch
            try:
                for index, value in series.iteritems():
                    d = {"index": index, column: value}
                    run.log_row(f"{slug}-{column}", description, **d)
            except AzureMLException:
                # This is a problem, but not so big that we should crash and burn
                logger.warning(
                    f"{slug} Failed to log {column} to azureML experiment logging, skipping (AzureMLException)"
                )
            except Exception as e:
                logger.warning(
                    f"{slug} Failed to log {column} to azureML experiment logging (Exception)"
                )
                raise e

    @staticmethod
    def log_figure_matplotlib(figure: plt.Figure, slug: str) -> None:
        run = Run.get_context()
        run.log_image(slug, plot=figure)

    @staticmethod
    def log_figure_plotly(figure: go.Figure, slug: str) -> None:
        """
        AzureML does not currently support direct logging of plotly objects, so save to file and log the file
        :param figure: plotly figure
        :param slug: first part of name of file (a unique ID is added after, so you can use the same slug multiple times)
        :return: None
        """
        run = Run.get_context()
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".png", dir=".", prefix=f"{slug}-"
        ) as png_file:
            figure.write_image(png_file.name, scale=3)
            logger.debug(f"Wrote plotly figure to temporary file {png_file.name}")
            run.log_image(slug, path=png_file.name)
