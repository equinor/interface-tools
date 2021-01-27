from enum import Enum
from typing import Any, Union

import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib import pyplot as plt

from interface_tools.infrastructure.experiment_logger import ExperimentLoggerAzureml


class MyEnum(str, Enum):
    enum_representation_1 = "string-representation-1"


@pytest.mark.parametrize(
    "key_in, value_in, key_out_expected, value_out_expected",
    [
        ("int", 0, "int", 0),
        ("float", 0.123, "float", 0.123),
        ("str", "test", "str", None),
        ("numpy float", np.float64(1.23), "numpy float", 1.23),
        (
            "numpy float",
            np.float32(1.23),
            "numpy float",
            None,
        ),  # float32 is not handled
        (MyEnum.enum_representation_1, 123, MyEnum.enum_representation_1.value, 123),
    ],
)
def test_ExperimentLoggerAzureml_ShouldCastScalarsToCorrectTypes_WhenCalled(
    key_in: Union[str, Enum],
    value_in: Any,
    key_out_expected: str,
    value_out_expected: Any,
) -> None:

    # Arrange
    logger = ExperimentLoggerAzureml()

    # Act
    key_out, value_out = logger._prepare_scalar(key_in, value_in)

    # Assert
    if value_out_expected is not None:
        assert key_out == key_out_expected
        assert value_out == value_out_expected
    else:
        assert key_out is None


@pytest.mark.parametrize(
    "key_in, prefix_in, key_expected",
    [
        ("A", None, "A"),
        ("A", "PREFIX", "PREFIX-A"),
    ],
)
def test_ExperimentLoggerAzureml_ShouldAddPrefixToKey_WhenCalled(
    key_in: str, prefix_in: str, key_expected: str
) -> None:

    # Arrange
    logger = ExperimentLoggerAzureml()

    # Act
    key_out, value_out = logger._prepare_scalar(key_in, 1.234, prefix_in)

    # Assert
    if prefix_in is None:
        assert key_out == key_in
    else:
        assert key_out == f"{prefix_in}-{key_in}"


def test_ExperimentLoggerAzureml_ShouldLogMatplotlibFigureWithoutError_WhenCalled() -> None:

    plt.figure()
    plt.scatter(x=[0, 1, 2, 3], y=[1, 2, 3, 4])

    logger = ExperimentLoggerAzureml()

    # Act
    logger.log_figure_matplotlib(figure=plt.gcf(), slug="test")

    # assert
    # Test will pass so long as there is no error thrown
    assert True


def test_ExperimentLoggerAzureml_ShouldLogPlotlyFigureWithoutError_WhenCalled() -> None:

    # Arrange
    fig = go.Figure(
        data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
        layout=go.Layout(
            title=go.layout.Title(text="A Figure Specified By A Graph Object")
        ),
    )

    logger = ExperimentLoggerAzureml()

    # Act
    logger.log_figure_plotly(figure=fig, slug="test")

    # assert
    # Test will pass so long as there is no error thrown
    assert True
