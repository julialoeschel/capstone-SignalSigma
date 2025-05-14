# ---
# description: (Additional) Configurations required for legacy code/notebooks.
# ---

import os
import pandas as pd
import json
from typing import List
from signal_sigma.config.cfg import *

# Canonical name of index column
IDX = "idx"

# Relative paths for data storage
DATA_STOCKS_DIR_RELPATH = "stocks"
DATA_FED_CEI_RELPATH = os.path.join("fed", "combined-economic-indicators.csv")
DATA_YF_MIF_RELPATH = os.path.join("yf", "macro-indicators-full.csv")


# Create Cartesian Product of Strings


def cartprod(*strss: List[str | List[str]]) -> str | List[str]:
    if len(strss) == 0:
        return []
    elif len(strss) == 1:
        return strss[0]
    else:
        head, tail = strss[0], cartprod(*strss[1:])
        is_head_str = isinstance(head, str)
        is_tail_str = isinstance(tail, str)
        if is_head_str and is_tail_str:
            return f"{head}_{tail}"
        else:
            head = [head] if is_head_str else head
            tail = [tail] if is_tail_str else tail
            return [f"{h}_{t}" for h in head for t in tail]


# ---


def store_df_as_csv(
    df: pd.DataFrame,
    relpath: str,
    version: int,
    root: str = DATA_PATH,
) -> None:
    csvpath = os.path.join(root, str(version), relpath)
    csvdir, _ = os.path.split(csvpath)
    os.makedirs(csvdir, exist_ok=True)
    df.to_csv(csvpath)
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    jsonpath = csvpath.replace(".csv", ".json")
    with open(jsonpath, "w") as fh:
        json.dump(dtypes, fh, indent=4)


def load_df_from_csv(
    csvpath_rel: str,
    nb_number: int,
    root: str = DATA_PATH,
) -> pd.DataFrame:
    csvpath = os.path.join(root, str(nb_number - 1), csvpath_rel)
    df = pd.read_csv(csvpath, index_col=IDX)
    jsonpath = csvpath.replace(".csv", ".json")
    with open(jsonpath, "r") as fh:
        dtypes = json.load(fh)
    df = df.astype(dtypes)
    return df
