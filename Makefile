PYTHON ?= .venv/bin/python
VENV_PYTHON ?= python3.12
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export PYTHONPATH := $(CURDIR)/src
# Use one OpenMP runtime for LightGBM and PyTorch on macOS. No duplicate-runtime bypass.
ifeq ($(shell uname -s),Darwin)
export DYLD_LIBRARY_PATH := $(CURDIR)/.venv/lib/python3.12/site-packages/torch/lib
endif
export MPLCONFIGDIR := $(CURDIR)/work/matplotlib

.PHONY: data analysis test lint reproduce setup backtest reliability report verify
setup:
	"$(VENV_PYTHON)" -m venv .venv
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install --no-deps --no-build-isolation .
	$(PYTHON) -m gridcast.learning
data:
	$(PYTHON) -m gridcast.data --root .
analysis: data
	$(PYTHON) -m gridcast.eda --root .
	$(PYTHON) -m gridcast.backtesting --root .
	$(PYTHON) -m gridcast.reliability --root .
	$(PYTHON) -m gridcast.reporting --root .
backtest:
	$(PYTHON) -m gridcast.backtesting --root .
test:
	$(PYTHON) -m pytest -q
lint:
	.venv/bin/ruff check src tests
reproduce:
	$(PYTHON) -m gridcast.audit --root .
reliability:
	$(PYTHON) -m gridcast.reliability --root .

report:
	$(PYTHON) -m gridcast.reporting --root .

verify:
	$(PYTHON) -m gridcast.audit --root . --saved-only
