VENV ?= venv313
PY ?= python3
VENV_PY := $(VENV)/bin/python
PIP := $(VENV_PY) -m pip

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "the-machine v2.0 - comandos disponibles:"
	@echo "  make setup      Crea el entorno virtual e instala las dependencias"
	@echo "  make start      Crea el entorno (si falta), instala paquetes y arranca main.py"
	@echo "                  Uso: make start          (modo hand)"
	@echo "                       make start ARGS=position"
	@echo "                       make start ARGS=music"
	@echo "  make test       Ejecuta la suite de pruebas con pytest"
	@echo "  make lint       Analisis estatico con ruff"
	@echo "  make format     Formatea el codigo con black"
	@echo "  make clean      Elimina el entorno virtual y las caches"

.PHONY: venv
venv:
	@test -x $(VENV_PY) || $(PY) -m venv $(VENV)

.PHONY: install
install: venv ## Instala las dependencias necesarias en el entorno
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

.PHONY: setup
setup: install ## Crea el entorno virtual e instala los paquetes necesarios

.PHONY: start
start: setup ## Inicia la maquina (entorno + paquetes + main.py)
	$(VENV_PY) main.py $(ARGS)

.PHONY: test
test: ## Ejecuta las pruebas del proyecto
	$(VENV_PY) -m pytest -q

.PHONY: lint
lint: ## Analiza el codigo con ruff
	$(VENV_PY) -m ruff check .

.PHONY: format
format: ## Formatea el codigo con black
	$(VENV_PY) -m black .

.PHONY: clean
clean: ## Elimina el entorno virtual y las caches
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache .coverage