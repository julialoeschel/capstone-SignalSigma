# ---
# title: Σιγναλ Σιγμα Makefile
# ---

# Global Variables
PYTHON_VERSION := 3.11.3
VENV := .venv
TARGETS := help basic-unix basic-win pre-commit-unix pre-commit-win

# Descriptors that should be treated as targets and not as files.
.PHONY: $(TARGETS)

# Target Definitions

help:
	@echo
	@echo "  ⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡"
	@echo "  ⟡ Σιγναλ Σιγμα ⟡"
	@echo "  ⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡"
	@echo
	@echo "  Install options:"
	@echo "  ⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡"
	@echo "    make basic-unix      - Set up virtual environment and dependencies on macOS/Linux"
	@echo "    make basic-win       - Set up virtual environment and dependencies on Windows (PowerShell)"
	@echo "    make pre-commit-unix - Set up pre-commit hooks on macOS/Linux"
	@echo "    make pre-commit-win  - Set up pre-commit hooks on Windows (PowerShell)"
	@echo

basic-unix:
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt

basic-win:
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	.\$(VENV)\Scripts\python.exe -m pip install --upgrade pip
	.\$(VENV)\Scripts\python.exe -m pip install -r requirements.txt

pre-commit-unix:
	pip install pre-commit
	pre-commit install

pre-commit-win:
	pip install pre-commit
	pre-commit install
