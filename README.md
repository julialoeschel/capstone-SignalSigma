# ⟡⟡⟡ Σιγναλ Σιγμα ⟡⟡⟡

## Synopsis

**TODO** Write this paragraph

## Requirements

- Python 3.11.3
- pyenv
<!-- - Node.js -->

And additionally, as usual, the modules to be installed for the virtual environment are listed in `requirements.txt`.

## Setup Options

### Basic

For exploring the repository, the basic installation option is the recommended choice.

#### Basic -- macOS/Linux

Run either `make basic-unix` or execute:

```shell
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Basic -- Windows (PowerShell)

Run either `make basic-win` or execute:

```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Extra

For a smoother "committing experience" when contributing, it is recommended to install additionally the [pre-commit framework](https://pre-commit.com).

#### Extra -- macOS/Linux

Once the virtual environment is activated, run either `make unix-extra` or execute:

```shell
pip install pre-commit
pre-commit install
```

#### Extra -- Windows (PowerShell)

Once the virtual environment is activated, run either `make win-extra` or execute:

```powershell
pip install pre-commit
pre-commit install
```
