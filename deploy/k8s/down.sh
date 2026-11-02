#!/usr/bin/env bash
# Remove the local cluster and everything in it. Run with `make k8s-down`.
set -euo pipefail
kind delete cluster --name docs-rag
