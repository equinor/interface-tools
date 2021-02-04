import json
import logging
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from interface_tools.utils.formatting import datetime_columns_to_string


class DataFrameRunner:
    def __init__(self, run_fn: Callable[[Dict, pd.DataFrame], pd.DataFrame]) -> None:
        self.run_fn = run_fn

    def run(self, artifacts: Dict, data: str) -> List[Dict]:
        logger = logging.getLogger()

        df = pd.DataFrame(json.loads(data))

        logger.info(f"Prediction requested with input data of {len(df)} rows")

        df_out = self.run_fn(artifacts, df)

        # 1. The `to_json()` function will crash if there are multiple rows with same index value.
        # Since the timestamp is set to be the index, this can error occur and therefore we need
        # to reset the index to ensure that it is unique.
        # 2. The to_json function does not convert datetime columns to a human readable format, so we have
        # convert them to strings before calling the json formatter.
        df_out = datetime_columns_to_string(df_out.reset_index())
        df_out = df_out.fillna("NaN")
        df_out = df_out.replace(-np.Inf, "-Inf")
        df_out = df_out.replace(np.Inf, "Inf")
        # TODO discard columns we dont need otherwise this dataframe may be HUGE
        return df_out.to_dict("records")
