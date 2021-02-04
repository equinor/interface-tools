import pandas as pd


def datetime_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Converts all columns and/or index of a date type to string.

    :param df: Dataframe with datetime index and/or columns
    :type df: pd.DataFrame
    :return: Dataframe with dates converted to string
    :rtype: pd.DataFrame
    """
    df = df.copy()
    if type(df.index) == pd.DatetimeIndex:
        df.index = df.index.astype(str)
    for col in df.select_dtypes(
        include=["datetime64", "timedelta64", "datetimetz"]
    ).columns:
        df[col] = df[col].astype("str")
    return df
