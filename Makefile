PYTHON   ?= python3.11
VENV     ?= .venv
PY       := $(VENV)/bin/python
MODEL    ?= nvidia/parakeet-tdt-0.6b-v3
LIMIT    ?= 300
THREADS  ?= 4
DATASETS ?= librispeech edacc fleurs_de
VARIANTS ?= torch onnx-fp32 onnx-int8

.PHONY: all setup export quantize data check bench report test lint clean

all: export quantize data check bench report

setup: $(VENV)/.installed

$(VENV)/.installed: requirements-dev.txt pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-dev.txt -e .
	touch $@

export: setup
	$(PY) scripts/export_onnx.py --model $(MODEL) --out models

quantize: setup
	$(PY) scripts/quantize.py --src models/fp32 --dst models/int8

data: setup
	$(PY) scripts/prepare_data.py --limit $(LIMIT) --datasets $(DATASETS)

check: setup
	$(PY) scripts/check.py --model $(MODEL) --threads $(THREADS)

# One process per variant so peak memory is measured per variant.
bench: setup
	@for v in $(VARIANTS); do \
		$(PY) scripts/bench.py --variant $$v --model $(MODEL) --threads $(THREADS) --datasets $(DATASETS) || exit 1; \
	done

report: setup
	$(PY) scripts/report.py

test: setup
	$(PY) -m pytest -q

lint: setup
	$(VENV)/bin/ruff check src scripts tests
	$(VENV)/bin/ruff format --check src scripts tests

clean:
	rm -rf results
