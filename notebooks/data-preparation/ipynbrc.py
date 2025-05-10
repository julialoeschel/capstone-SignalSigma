# ---
# description: This file contains "global variables". It will be run at the beginning of each notebook.
# ---

# ---

import os
from shutil import rmtree
from typing import List
import json
import pandas as pd
import matplotlib.pyplot as plt
import subprocess


# ---

# PATHS

PATH_REPO = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True,
    text=True,
).stdout.strip()

PATH_DATA = os.path.join(PATH_REPO, "data")
PATH_PLOTS = os.path.join(PATH_REPO, "plots")

# ---

# Single stocks

STOCK_TICKERS_ = {
    "AAPL": "Apple",
    "AMD": "AMD",
    "AMZN": "Amazon",
    "AVGO": "Broadcom",
    "CRM": "Salesforce",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
}

STOCK_TICKERS = list(STOCK_TICKERS_.keys())

DIR_DATA_STOCKS = "stocks"

# ---

# FED

RELPATH_DATA_FED_CEI = os.path.join("fed", "combined-economic-indicators.csv")

# ---

# FED

RELPATH_DATA_YF_MIF = os.path.join("yf", "macro-indicators-full.csv")

# ---

# Storing and loading


def store_df_as_csv(
    df: pd.DataFrame,
    relpath: str,
    version: int,
    root: str = PATH_DATA,
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
    root: str = PATH_DATA,
) -> pd.DataFrame:
    csvpath = os.path.join(root, str(nb_number - 1), csvpath_rel)
    df = pd.read_csv(csvpath, index_col=IDX)
    jsonpath = csvpath.replace(".csv", ".json")
    with open(jsonpath, "r") as fh:
        dtypes = json.load(fh)
    df = df.astype(dtypes)
    return df


# ---

# Reset directories which are not tracked by git


def reset_directory(dirpath: str) -> None:
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    for item in os.listdir(dirpath):
        item_path = os.path.join(dirpath, item)
        if item != ".gitkeep":
            if os.path.isdir(item_path):
                rmtree(item_path)
            else:
                os.remove(item_path)


# ---

# Cartesian product of strings


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

# Matplotlib style
# PLT_STYLE = "seaborn-v0_8-darkgrid"
PLT_STYLE = "dark_background"
PLT_STYLE = "fast"

try:
    plt.style.use(PLT_STYLE)
except:
    print(f"Could not load matplotlib style {PLT_STYLE}.")
    print("Classic style will be used.")

plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["axes.labelcolor"] = "white"

plt.rcParams["figure.dpi"] = 600
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["savefig.format"] = "svg"
plt.rcParams["savefig.dpi"] = 600

plt.rcParams["grid.color"] = "gray"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

plt.rcParams["legend.loc"] = "upper right"
plt.rcParams["legend.frameon"] = False
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 8


def save_plot(
    corename: str,
    path: str = PATH_PLOTS,
    extension: str = "svg",
) -> None:
    filepath = os.path.join(path, f"{corename}.{extension}")
    plt.savefig(filepath, bbox_inches="tight")


# ---

# Random seed
RSEED = 42

# ---

# Canonical name of index column
IDX = "idx"

# ---
