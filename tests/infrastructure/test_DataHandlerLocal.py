import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Generator, Tuple

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from interface_tools.infrastructure.data_definitions import FileType, StorageType
from interface_tools.infrastructure.DataHandler import DataHandler


@pytest.fixture
def local_pickle_file_at_project_root() -> Generator:
    FILENAME = "tmp.local_pickle_file_at_project_root"
    base_path = Path(__file__).parent.parent.parent
    file_path = base_path / f"{FILENAME}.pkl"
    df = pd.DataFrame(data={"data": [1, 2, 3]})
    with open(file_path, "wb") as file:
        pickle.dump(df, file)
    yield (base_path, FILENAME, df)
    os.remove(file_path)


def test_DataHandlerLocal_ShouldLoadPickleFile_WithNoPath(
    local_pickle_file_at_project_root: Tuple,
) -> None:
    base_path, filename, df = local_pickle_file_at_project_root

    config = {
        "storage_type": StorageType.LOCAL,
        "file_type": FileType.PICKLE,
        "name": filename,
    }

    df_ = DataHandler[pd.DataFrame]().load(config, base_path=base_path)

    assert df_.equals(df)


def test_DataHandlerLocal_ShouldLoadPickleFile_WithAbsolutePath(
    local_pickle_file_at_project_root: Tuple,
) -> None:
    base_path, filename, df = local_pickle_file_at_project_root

    config = {
        "storage_type": StorageType.LOCAL,
        "file_type": FileType.PICKLE,
        "name": base_path / filename,
    }

    df_ = DataHandler[pd.DataFrame]().load(config, base_path=Path(""))

    assert df_.equals(df)


def test_DataHandlerLocal_ShouldLoadPickleFile_WithRelativePath(
    local_pickle_file_at_project_root: Tuple,
) -> None:
    base_path, filename, df = local_pickle_file_at_project_root

    config = {
        "storage_type": StorageType.LOCAL,
        "file_type": FileType.PICKLE,
        "relative_path": base_path.name,
        "name": filename,
    }

    df_ = DataHandler[pd.DataFrame]().load(config, base_path=base_path.parent)

    assert df_.equals(df)


@pytest.fixture
def setup_teardown_tempdir():
    base_path = Path(tempfile.TemporaryDirectory().name)
    yield base_path
    print(f"Removing {base_path}")
    for root, dirs, files in os.walk(base_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
    os.rmdir(base_path)


def test_DataHandlerLocal_ShouldSaveMatplotlibFigAsPng_WhenCalled(
    setup_teardown_tempdir,
) -> None:

    base_path = setup_teardown_tempdir

    filename = "mytest"
    config = {
        "storage_type": StorageType.LOCAL,
        "file_type": FileType.MATPLOTLIB_PNG,
        "relative_path": base_path.name,
        "name": filename,
    }

    fig, ax = plt.subplots()

    DataHandler[Any]().save(fig, config=config, base_path=base_path.parent)

    assert os.path.isfile(Path(base_path) / f"{filename}.png")
