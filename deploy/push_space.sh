#!/usr/bin/env bash
# Publish the committed HEAD to a Hugging Face Space (Docker SDK).
#
#   deploy/push_space.sh <hf-username>/<space-name>
#
# One-time setup: create the Space at https://huggingface.co/new-space (SDK: Docker, visibility: Public)
# and create a write token at https://huggingface.co/settings/tokens. Export it as HF_TOKEN.
#
# The Space gets its own history: this script exports HEAD, prepends the YAML header Spaces needs to
# the README, and force-pushes that snapshot, so the GitHub repo and README stay untouched.
set -euo pipefail

SPACE="${1:?usage: deploy/push_space.sh <hf-username>/<space-name>}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face write token}"

root="$(git rev-parse --show-toplevel)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git -C "$root" archive HEAD | tar -x -C "$work"
{
  printf -- '---\ntitle: DocIntel\nemoji: 📄\ncolorFrom: indigo\ncolorTo: purple\nsdk: docker\napp_port: 7860\npinned: false\n---\n\n'
  cat "$work/README.md"
} > "$work/README.new" && mv "$work/README.new" "$work/README.md"

cd "$work"
git init -q -b main
git add -A
git -c user.name="$(git -C "$root" config user.name)" -c user.email="$(git -C "$root" config user.email)" \
    commit -q -m "Deploy $(git -C "$root" rev-parse --short HEAD)"
# The token is passed via a one-off header, never written to the remote URL or git config.
auth="$(printf 'user:%s' "$HF_TOKEN" | base64 | tr -d '\n')"
git -c http.extraHeader="Authorization: Basic $auth" push -f "https://huggingface.co/spaces/$SPACE" main
echo "Pushed. Build logs: https://huggingface.co/spaces/$SPACE"
