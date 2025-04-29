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

For exploring the repository, the basic installation option is the right choice.

#### Basic -- macOS/Linux

Run either `make basic-unix` or the following lines:

```shell
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Basic -- Windows (PowerShell)

Run either `make basic-win` or the following lines:

```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Extra

For a smoother "committing experience" when contributing, it is recommended to install additionally the [pre-commit framework](https://pre-commit.com). On macOS, Linux and Windows (PowerShell), you can manually execute the following lines:

#### Extra -- macOS/Linux

In the activated virtual environment, run either `make unix-extra` or the following lines:

```shell
pip install pre-commit
pre-commit install
```

#### Extra -- Windows (PowerShell)

In the activated virtual environment, run either `make win-extra` or the following lines:

```powershell
pip install pre-commit
pre-commit install
```
