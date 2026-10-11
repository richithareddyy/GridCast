PYTHON ?= .venv/bin/python
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export PYTHONPATH := $(CURDIR)/src
export MPLCONFIGDIR := $(CURDIR)/work/matplotlib

.PHONY: data analysis test lint reproduce setup backtest
setup:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install --no-deps --no-build-isolation .
data:
	$(PYTHON) -m gridcast.data --root .
analysis: data
	$(PYTHON) -m gridcast.eda --root .
	$(PYTHON) -m gridcast.backtesting --root .
backtest:
	$(PYTHON) -m gridcast.backtesting --root .
test:
	$(PYTHON) -m pytest -q
lint:
	.venv/bin/ruff check src tests
reproduce:
	$(PYTHON) -m gridcast.audit --root .
