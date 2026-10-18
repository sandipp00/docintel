#!/usr/bin/env bash
# Deploy DocIntel to Google Cloud Run.
#
#   deploy/cloud_run.sh <gcp-project-id> [region]
#
# One-time setup: a Google Cloud project with billing enabled, the gcloud CLI, and `gcloud auth login`.
# Cloud Build builds the image from the Dockerfile in the cloud, so Docker isn't needed locally.
#
# Deliberate choices:
#   --memory 2Gi      PyTorch plus the embedding model need well over Cloud Run's 512Mi default.
#   --max-instances 1 Documents and the FAISS index live in one process's memory; a second instance
#                     would serve visitors a store that doesn't have their uploads in it.
#   --min-instances 0 Scales to zero so an idle demo costs nothing (the price is a slow first request).
#   --timeout 300     OCR on a multi-page scan can take well over the 60s default.
#   The LLM stays off unless you pass a key; see the note at the end.
set -euo pipefail

PROJECT="${1:?usage: deploy/cloud_run.sh <gcp-project-id> [region]}"
REGION="${2:-us-central1}"
SERVICE=docintel

root="$(git rev-parse --show-toplevel)"
cd "$root"

echo "==> Project $PROJECT, region $REGION"
gcloud config set project "$PROJECT" --quiet

echo "==> Enabling the APIs Cloud Run needs (no-op if already on)"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --quiet

echo "==> Building and deploying (first run takes ~10 minutes: the image is ~3 GB)"
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 1 \
  --min-instances 0 \
  --timeout 300 \
  --quiet

url="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo
echo "Live at: $url"
echo
echo "The AI is off by default, so nothing public can spend your API quota. To turn it on:"
echo "  gcloud run services update $SERVICE --region $REGION \\"
echo "      --update-env-vars LLM_PROVIDER=gemini --update-secrets GEMINI_API_KEY=docintel-gemini:latest"
echo "(store the key first: printf %s \"\$KEY\" | gcloud secrets create docintel-gemini --data-file=-)"
