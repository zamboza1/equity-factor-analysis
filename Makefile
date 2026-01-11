# Makefile shortcuts for radar tool

.PHONY: help analyze test install clean

# Default target
help:
	@echo "Stock Factor Analysis"
	@echo ""
	@echo "Available commands:"
	@echo "  make analyze TICKER=AAPL          - Run full analysis (default: AAPL)"
	@echo "  make analyze-msft                - Quick: analyze MSFT"
	@echo "  make analyze-spy                 - Quick: analyze SPY"
	@echo "  make event-study                 - Run CPI event study"
	@echo "  make test                        - Run all tests"
	@echo "  make install                     - Install package"
	@echo "  make clean                       - Clean cache and reports"
	@echo ""

# Full analysis
analyze:
	@python -m radar.cli analyze $(or $(TICKER),AAPL) --start 2023-01-01 --end $$(date +%Y-%m-%d)

# Quick shortcuts for common tickers
analyze-aapl:
	@python -m radar.cli analyze AAPL --start 2023-01-01 --end $$(date +%Y-%m-%d)

analyze-msft:
	@python -m radar.cli analyze MSFT --start 2023-01-01 --end $$(date +%Y-%m-%d)

analyze-spy:
	@python -m radar.cli analyze SPY --start 2023-01-01 --end $$(date +%Y-%m-%d)

analyze-googl:
	@python -m radar.cli analyze GOOGL --start 2023-01-01 --end $$(date +%Y-%m-%d)

# Event study
event-study:
	@python -m radar.cli event-study --event CPI --assets SPY,TLT

# Tests
test:
	@pytest tests/ -v

# Install
install:
	@pip install -e .

# Clean
clean:
	@rm -rf .cache reports/*.md reports/assets/*.png
	@echo "Cleaned cache and reports"

