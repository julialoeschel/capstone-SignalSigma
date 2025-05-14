# ---
# title: Σιγναλ Σιγμα Makefile
# ---

# Global Variables
PYTHON_VERSION := 3.11.3
VENV := .venv
TARGETS := help basic-unix basic-win dev-unix dev-win clear-unix clear-win reset-unix reset-win
BOLD_WHITE := \033[1;37m
RESET := \033[0m

# Descriptors that should be treated as targets and not as files.
.PHONY: $(TARGETS)

# Target Definitions

help:
	@echo
	@echo "    $(BOLD_WHITE)⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡$(RESET)"
	@echo "    $(BOLD_WHITE)⟡ Σιγναλ Σιγμα ⟡$(RESET)"
	@echo "    $(BOLD_WHITE)⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡$(RESET)"
	@echo
	@echo "    $(BOLD_WHITE)Setup:$(RESET)"
	@echo "    make basic-unix      - Set up virtual environment and dependencies on macOS/Linux"
	@echo "    make basic-win       - Set up virtual environment and dependencies on Windows (PowerShell)"
	@echo "    make dev-unix        - Set up development environment and pre-commit hooks on macOS/Linux"
	@echo "    make dev-win         - Set up development environment and pre-commit hooks on Windows (PowerShell)"
	@echo 
	@echo "    $(BOLD_WHITE)Clean-Up:$(RESET)"
	@echo "    make clear-unix      - Clear build artifacts in data, logs, plots on macOS/Linux"
	@echo "    make clear-win       - Clear build artifacts in data, logs, plots on Windows (PowerShell)"
	@echo "    make reset-unix      - Clear build artifacts and remove virtual environment on macOS/Linux"
	@echo "    make reset-win       - Clear build artifacts and remove virtual environment on Windows (PowerShell)"
	@echo
	@echo "    $(BOLD_WHITE)Important Make Flags:$(RESET)"
	@echo "    -n                   - Dry-run (print commands without running them)"
	@echo "    -s                   - Silent mode (don't print executed commands)"
	@echo "    --debug[=b|v|a]      - Debug info (b=basic [default], v=verbose, a=all)"
	@echo

basic-unix:
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e .

basic-win:
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	.\$(VENV)\Scripts\python.exe -m pip install --upgrade pip
	.\$(VENV)\Scripts\python.exe -m pip install -e .

dev-unix:
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e .[dev]
	$(VENV)/bin/pre-commit install

dev-win:
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	.\$(VENV)\Scripts\python.exe -m pip install --upgrade pip
	.\$(VENV)\Scripts\python.exe -m pip install -e .[dev]
	.\$(VENV)\Scripts\pre-commit.exe install

clear-unix:
	find data -mindepth 1 ! -name '.gitkeep' -delete
	find logs -mindepth 1 ! -name '.gitkeep' -delete
	find plots -mindepth 1 ! -name '.gitkeep' -delete

clear-win:
	if exist data (for %%f in (data\*) do if /I not "%%~nxf"==".gitkeep" del "%%f")
	if exist logs (for %%f in (logs\*) do if /I not "%%~nxf"==".gitkeep" del "%%f")
	if exist plots (for %%f in (plots\*) do if /I not "%%~nxf"==".gitkeep" del "%%f")

reset-unix:
	$(MAKE) clear-unix
	rm -rf $(VENV)
	pyenv local --unset

reset-win:
	$(MAKE) clear-win
	if exist $(VENV) rmdir /s /q $(VENV)
	pyenv local --unset
