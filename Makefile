PYTHON ?= python3
TICKER ?= AAPL
START ?= 2024-01-01
END ?= $(shell date +%Y-%m-%d)
PORT ?= 8000

.PHONY: help analyze analyze-aapl analyze-msft analyze-spy analyze-googl event-study web test install clean

help:
	@echo "Equity Factor Analysis"
	@echo "make analyze TICKER=AAPL START=2024-01-01 END=2025-01-01"
	@echo "make web PORT=8000 | make event-study | make test | make install"
	@echo "make clean deletes local cached downloads and generated reports"

analyze:
	$(PYTHON) -m backend.cli analyze $(TICKER) --start $(START) --end $(END)

analyze-aapl analyze-msft analyze-spy analyze-googl:
	$(MAKE) analyze TICKER=$(shell echo $(@:analyze-%=%) | tr '[:lower:]' '[:upper:]')

event-study:
	$(PYTHON) -m backend.cli event-study --event CPI --assets SPY,TLT

web:
	$(PYTHON) -m backend.web --port $(PORT)

test:
	$(PYTHON) -m pytest tests -q

install:
	$(PYTHON) -m pip install -e ".[dev]"

clean:
	rm -rf .cache reports
