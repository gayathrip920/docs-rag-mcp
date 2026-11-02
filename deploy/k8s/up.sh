#!/usr/bin/env bash
# Start a local Kubernetes cluster (kind), build the images, and deploy the app.
# Needs: docker, kind, kubectl. Run from the repo root with `make k8s-up`.
set -euo pipefail

CLUSTER=docs-rag
NAMESPACE=docs-rag
# The file that holds DATABASE_URL. Use the read-only login here (see deploy/sql).
SECRET_ENV_FILE="${SECRET_ENV_FILE:-.env.reader}"

if [ ! -f "$SECRET_ENV_FILE" ]; then
  echo "Missing $SECRET_ENV_FILE. Put your DATABASE_URL in it, or set SECRET_ENV_FILE." >&2
  exit 1
fi
DATABASE_URL="$(grep -E '^DATABASE_URL=' "$SECRET_ENV_FILE" | head -n1 | cut -d= -f2-)"
if [ -z "$DATABASE_URL" ]; then
  echo "No DATABASE_URL found in $SECRET_ENV_FILE." >&2
  exit 1
fi

if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER"
fi
kubectl config use-context "kind-$CLUSTER" >/dev/null

docker build -f deploy/docker/Dockerfile.api -t docs-rag-api:local .
docker build -f deploy/docker/Dockerfile.frontend -t docs-rag-web:local .
kind load docker-image docs-rag-api:local docs-rag-web:local --name "$CLUSTER"

kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
# The secret is made here from your local file. It is never written into the repo.
kubectl -n "$NAMESPACE" create secret generic docs-rag-secrets \
  --from-literal=DATABASE_URL="$DATABASE_URL" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f deploy/k8s/api-deployment.yaml -f deploy/k8s/api-service.yaml
kubectl apply -f deploy/k8s/frontend-deployment.yaml -f deploy/k8s/frontend-service.yaml

kubectl -n "$NAMESPACE" rollout status deployment/api --timeout=300s
kubectl -n "$NAMESPACE" rollout status deployment/web --timeout=120s
kubectl -n "$NAMESPACE" get pods

echo
echo "Open the page with:  kubectl -n $NAMESPACE port-forward svc/web 3000:3000"
echo "Then visit http://localhost:3000"
