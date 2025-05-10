# Notebooks

## Annotated TOC

### 1. [Data Retrieval](./1--data-collecting.ipynb)

**TODO** Explain three fetching sources

- Stocks from yahooFinance.
- MacroIndicators from yahooFinance.
- FED

For fetching the data from yahooFinance (`yfinance`) and FED (`Fred`), API credentials are necessary. You need a `.env` at the root of the repository directory containing these credentials, i.d. the content of `.env` should be

``` python
FINANCE_API_KEY = "foobarbaz"
FRED_API_KEY = "42"
```

### 2. [Data Inspection](./2--data-inspection.ipynb)

### 3. [Merge](./3--data-merging.ipynb)

### 4. [Feature Engineering](./4--feature-engineering.ipynb)

### 5. [Time Series Decomposition](./5--ts-decomposition.ipynb)

## Remarks

The data fetched and processed in the Notebooks will be collected in the `data` directory. For the first notebooks where we collect and process the data there are corresponding subdirectories of `data` storing the data frames as they are at the end of each notebook, respectively, e.g. subdirectory `42` contains the data frames that result from running notebook `41--foo.ipynp` and so on.

**TODO** Explain structure of data folder
