SHELL := /bin/bash
ROOT := $(shell pwd)
IMAGE := whatsapp-agent:local
PORT ?= 8001

.PHONY: help
help:
	@echo "Targets: build, run, docker-up, docker-down, logs, lint, test, fmt, check"

.PHONY: build
build:
	docker build -t $(IMAGE) .

.PHONY: run
run:
	PORT=$(PORT) ENV=production docker run --rm -it -p $(PORT):8001 --env-file .env $(IMAGE)

.PHONY: docker-up
docker-up:
	docker compose up -d --build

.PHONY: docker-down
docker-down:
	docker compose down

.PHONY: logs
logs:
	docker compose logs -f agent

.PHONY: lint
lint:
	python -m black --check . && isort --check-only . && mypy --ignore-missing-imports server || true

.PHONY: fmt
fmt:
	python -m black . && isort .

.PHONY: test
test:
	pytest -q

.PHONY: check
check:
	bash ./check_deploy.sh || true
