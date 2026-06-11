#!/usr/bin/env bash
# Package the fine-tuned Qwen ranker into the model.tar.gz layout SageMaker
# expects, then (optionally) upload it to S3.
#
# SageMaker unpacks model.tar.gz into the container's model_dir and runs the
# handler in code/inference.py. Layout we build:
#
#   model.tar.gz
#   ├── adapter/           # PEFT/LoRA adapter      (from the trained checkpoint)
#   ├── tokenizer/         # tokenizer files         (from the trained checkpoint)
#   ├── score_head.pt      # ranking-head weights    (from the trained checkpoint)
#   └── code/
#       ├── inference.py    # the handler (this dir)
#       ├── requirements.txt # extra deps installed into the inference container
#       └── models/         # repo source: qwen.py + data.py (for identical scoring)
#
# Usage:
#   infra/sagemaker/package_model.sh <checkpoint_dir> [s3://bucket/key/model.tar.gz]
#
#   <checkpoint_dir>  a dir produced by `models.qwen` save_model — must contain
#                     adapter/, tokenizer/, and score_head.pt (i.e. RANKER_PATH).
#
# Requires: tar, and (for upload) the AWS CLI with credentials.
set -euo pipefail

CKPT="${1:?usage: package_model.sh <checkpoint_dir> [s3_uri]}"
S3_URI="${2:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

for f in adapter tokenizer score_head.pt; do
  [ -e "$CKPT/$f" ] || { echo "ERROR: $CKPT/$f not found — is this a ranker checkpoint?" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "[package] staging artifact from $CKPT"
cp -r "$CKPT/adapter"        "$STAGE/adapter"
cp -r "$CKPT/tokenizer"      "$STAGE/tokenizer"
cp    "$CKPT/score_head.pt"  "$STAGE/score_head.pt"

echo "[package] staging inference code"
mkdir -p "$STAGE/code/models"
cp "$HERE/code/inference.py"        "$STAGE/code/inference.py"
cp "$HERE/code/requirements.txt"    "$STAGE/code/requirements.txt"
# Only the two modules the handler's scoring path needs (qwen.py -> data.py).
cp "$REPO_ROOT/models/__init__.py"  "$STAGE/code/models/__init__.py"
cp "$REPO_ROOT/models/qwen.py"      "$STAGE/code/models/qwen.py"
cp "$REPO_ROOT/models/data.py"      "$STAGE/code/models/data.py"

# Build the tarball OUTSIDE the staging dir so tar doesn't archive its own
# output (which races as "file changed as we read it").
OUT="$(mktemp -d)/model.tar.gz"
echo "[package] building $OUT"
tar -C "$STAGE" -czf "$OUT" .

if [ -n "$S3_URI" ]; then
  echo "[package] uploading to $S3_URI"
  aws s3 cp "$OUT" "$S3_URI"
  echo "[package] done -> $S3_URI"
else
  FINAL="$REPO_ROOT/model.tar.gz"
  cp "$OUT" "$FINAL"
  echo "[package] done -> $FINAL (no S3 uri given; not uploaded)"
fi
