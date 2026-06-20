#!/usr/bin/env bash
set -euo pipefail
TAG="${1:-0.115.0}"
DEST="data/corpus/fastapi"
rm -rf "$DEST"
git clone --depth 1 --branch "$TAG" https://github.com/fastapi/fastapi.git "$DEST"
SHA="$(git -C "$DEST" rev-parse HEAD)"
echo "tag=$TAG sha=$SHA" > data/corpus/COMMIT.txt
echo "Pinned FastAPI $TAG @ $SHA"
echo "Source package: $DEST/fastapi"
