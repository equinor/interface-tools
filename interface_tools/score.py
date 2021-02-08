import logging
import os
from logging import Logger
from pathlib import Path
from typing import Any, Union

import joblib
from opencensus.ext.azure.log_exporter import AzureLogHandler

# This file is organized according to the suggestions in this tutorial:
# https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-convert-ml-experiment-to-production

artifacts = dict()

MODEL_NAME: str = "model.pkl"
AML_ENV: str = "AZUREML_MODEL_DIR"
APPINSIGHTS_ENV: str = "ENABLE_APPINSIGHTS"


def _setup_logging(enableAppInsights: Union[str, bool]) -> Logger:
    FORMAT = "%(asctime)-15s %(levelname)s %(module)s:%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    if enableAppInsights:
        # TODO: In deployment, the workspace config file is not available and therefore
        # accessing the key from the key vault in this way does not work. A quick fix
        # is to hard code the key, which is ok to do as the key is not considered to be very secret.

        # keyVault = Workspace.from_config().get_default_keyvault()
        # instrumentation_key = keyVault.get_secret(
        #     "aangdev-appinsights-instrumentationkey"
        # )
        instrumentation_key = "4162ac29-1ebb-45bb-9efb-8b77d1d034fa"

        logger.addHandler(
            AzureLogHandler(
                connection_string=f"InstrumentationKey={instrumentation_key}"
            )
        )
    return logger


def init() -> None:
    """
    Initialize 3-pipeline trading model, retrieving pipeline artifacts from pickle file
    Code is structured according to: https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-convert-ml-experiment-to-production
    :return:
    """
    global artifacts

    enableAppInsights = os.getenv(APPINSIGHTS_ENV) or False
    logger = _setup_logging(enableAppInsights)

    env = os.getenv(AML_ENV)
    if env:
        model_path = Path(env) / Path(MODEL_NAME)
    else:
        raise EnvironmentError("Azure ML model path environment variable is not set")

    artifacts = joblib.load(model_path)
    logger.info("Model has been loaded successfully")


def run(data: Any, request_headers: Any) -> str:
    """
    Make prediction using 3-pipeline trading model
    Code is structured according to: https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-convert-ml-experiment-to-production
    :param request_headers: request headers
    :return: dictionary containing 'result'
    """
    return artifacts["run"].run(artifacts, data)
