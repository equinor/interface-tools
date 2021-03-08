from enum import Enum


class StorageType(str, Enum):
    LOCAL = "local"
    AZ_ST_ACC = "azure_storage_account"
    AZ_ML_DS = "azure_ml_dataset"


class FileType(str, Enum):
    DATAFRAME = "dataframe"
    PICKLE = "pickle"
    HDF5 = "hdf5"
    NUMPY_ARR = "numpy_array"
    HTML = "html"
    PNG = "png"
    JSON = "json"
    MATPLOTLIB_PNG = "matplotlib_png"
    TENSORFLOW = "tensorflow"
    PARQUET = "PARQUET"
