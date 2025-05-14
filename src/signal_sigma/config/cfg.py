# ---
# description: This file contains global variables and configurations for the project.
# ---

# ---

import os
import subprocess
import matplotlib.pyplot as plt

# ---

# Random Seed
RSEED = 42

# ---

# Directories

# Root directory of the project

ROOT_PATH = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True,
    text=True,
).stdout.strip()


def join_with_root(*tokens: str, root: str = ROOT_PATH) -> str:
    path = os.path.join(root, *tokens)
    return path


DATA_PATH = join_with_root("data")
PLOTS_PATH = join_with_root("plots")
LOGS_PATH = join_with_root("logs")

# ---

# Stocks Plus Tickers

STOCKS = {
    "AAPL": "Apple Inc.",
    "ADBE": "Adobe Inc.",
    "AMD": "AMD",
    "AMZN": "Amazon.com Inc.",
    "AVGO": "Broadcom",
    "BAC": "Bank of America Corp.",
    "CRM": "Salesforce",
    "DIS": "The Walt Disney Co.",
    "GOOGL": "Alphabet Inc.",
    "HD": "Home Depot Inc.",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "MA": "Mastercard Inc.",
    "META": "Meta Platforms Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "PEP": "PepsiCo Inc.",
    "PFE": "Pfizer Inc.",
    "PG": "Procter & Gamble Co.",
    "TSLA": "Tesla Inc.",
    "UNH": "UnitedHealth Group Inc.",
    "V": "Visa Inc.",
    "WMT": "Walmart Inc.",
}

STOCK_TICKERS = list(STOCKS.keys())

# Macros Plus Tickers

# XXX: Which of MACROS and MACROS_ALT to retain?
MACROS = {
    "^VIX": "VIX_Index",
    "^GSPC": "SP500",
    "^DJI": "Dow_Jones",
    "^IXIC": "Nasdaq",
    "^RUT": "Russell_2000",
    "TNX": "US10Y_Yield",
    "TYX": "US30Y_Yield",
    "EURUSD=X": "EUR_USD",
    "GBPUSD=X": "GBP_USD",
    "JPY=X": "USD_JPY",
    "CL=F": "Crude_Oil_WTI",
    "BZ=F": "Brent_Crude",
    "GC=F": "Gold_Futures",
    "SI=F": "Silver_Futures",
    "HG=F": "Copper_Futures",
    "ZC=F": "Corn_Futures",
    "ZW=F": "Wheat_Futures",
    "^IRX": "US_13W_TBill",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "XMR-USD": "Monero",
    "ZEC-USD": "Zcash",
    "USDT-USD": "Tether",
    "USDC-USD": "USD_Coin",
    "BNB-USD": "Binance_Coin",
    "XRP-USD": "XRP",
}

MACRO_TICKERS = list(MACROS.keys())

MACROS_ALT = {
    # 🔹 Equity Indices
    "^GSPC": "sp500",  # Broad market (sentiment baseline)
    "^DJI": "dowjones",  # Industrial-heavy U.S. large caps
    "^IXIC": "nasdaq",  # Tech/growth-heavy index
    # 🔹 Volatility (Risk Appetite)
    "^VIX": "vix",  # Market fear index
    # 🔹 Interest Rates
    "^TNX": "10y_yield",  # Benchmark long-term rate
    "^IRX": "3mo_yield",  # Short-term rate (Fed influenced)
    # 🔹 Commodities
    "GC=F": "gold",  # Inflation hedge / risk-off asset
    "CL=F": "oil",  # Global demand & inflation driver
    # 🔹 Currency Strength
    "DX-Y.NYB": "dxy",  # Dollar Index
    # 🔹 ETFs (Market segments)
    "QQQ": "qqq",  # Tech exposure
    "XLK": "tech_etf",  # Broader technology sector
    "XLF": "financial_etf",  # Interest-sensitive stocks
    "XLE": "energy_etf",  # Energy inflation proxy
    "ARKK": "arkk",  # Speculative/growth
    "TLT": "longbond_etf",  # Long-term treasury bond
    "BND": "bond_market_etf",  # Total bond market
    # 🔹 Cryptocurrencies (speculative innovation)
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
}


# ---

# Matplotlib style

PLT_STYLE = "seaborn-v0_8-darkgrid"
PLT_STYLE = "dark_background"
PLT_STYLE = "fast"

try:
    plt.style.use(PLT_STYLE)
except:
    print("Could not load the specified matplotlib style:")
    print(PLT_STYLE)
    print("Default style will be used")

plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["axes.labelcolor"] = "white"

plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["savefig.format"] = "svg"
plt.rcParams["savefig.dpi"] = 300

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


# def save_plot(
#     plot_filename: str,
#     path: str = PLOTS_PATH,
# ) -> None:
#     filepath = os.path.join(path, plot_filename)
#     plt.savefig(filepath, bbox_inches="tight")
