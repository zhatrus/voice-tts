.PHONY: help env build up down restart update logs ps health voices shell clean

COMPOSE_FILE ?= docker-compose.yml
PORT ?= 8001

# Auto-detect the compose command: prefer the v2 plugin (`docker compose`),
# fall back to the standalone v1 binary (`docker-compose`).
DOCKER_COMPOSE := $(shell \
	if docker compose version >/dev/null 2>&1; then echo "docker compose"; \
	elif command -v docker-compose >/dev/null 2>&1; then echo "docker-compose"; \
	else echo ""; fi)
DC = $(DOCKER_COMPOSE) -f $(COMPOSE_FILE)

ifeq ($(strip $(DOCKER_COMPOSE)),)
$(warning Neither "docker compose" nor "docker-compose" was found in PATH. \
Install Docker, or run with sudo / add your user to the "docker" group.)
endif

help: ## Show this help
	@echo "Ukrainian Voice (TTS) — targets:"
	@echo "  make env       — create .env from .env.example (if missing)"
	@echo "  make build     — build the GPU image"
	@echo "  make up        — start in background"
	@echo "  make down      — stop"
	@echo "  make restart   — restart service"
	@echo "  make update    — rebuild from current code and recreate container"
	@echo "  make logs      — tail logs"
	@echo "  make ps        — container status"
	@echo "  make health    — curl /health"
	@echo "  make voices    — list available voices"
	@echo "  make shell     — shell into the container"
	@echo "  make clean     — stop and remove volumes + image"

env: ## Create .env from template
	@test -f .env || (cp .env.example .env && echo "Created .env — edit API_KEY / DEVICE / DEFAULT_VOICE")

build: ## Build image
	$(DC) build

up: ## Start service
	$(DC) up -d

down: ## Stop service
	$(DC) down

restart: ## Restart service
	$(DC) restart

update: ## Rebuild from current code and recreate the container
	$(DC) build
	$(DC) up -d --force-recreate

logs: ## Tail logs
	$(DC) logs -f --tail=200

ps: ## Show status
	$(DC) ps

health: ## Hit the health endpoint
	curl -fsS http://localhost:$(PORT)/health && echo

voices: ## List available voices
	curl -fsS http://localhost:$(PORT)/voices && echo

shell: ## Open a shell in the container
	$(DC) exec app bash

clean: ## Stop and remove volumes + image
	$(DC) down -v
	-docker image rm ukrainian-voice-app
