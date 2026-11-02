.PHONY: test lint docker-build compose-up k8s-up k8s-down

test:
	pytest

lint:
	ruff check . && ruff format --check .

docker-build:
	docker build -f deploy/docker/Dockerfile.api -t docs-rag-api:local .
	docker build -f deploy/docker/Dockerfile.frontend -t docs-rag-web:local .

compose-up:
	docker compose up --build

k8s-up:
	bash deploy/k8s/up.sh

k8s-down:
	bash deploy/k8s/down.sh
