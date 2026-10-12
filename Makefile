VENV ?= .venv

.PHONY: check
check:
	$(VENV)/bin/python -m pytest -q
