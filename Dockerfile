# syntax=docker/dockerfile:1

# CPU-only image for the DEPLOYED harvester (the scheduled Fargate task).
#
# It installs ONLY the `runtime` optional-dependency set from pyproject (no
# torch/peft/datasets/gradio/wandb), because the Qwen ranker is served by a
# SageMaker endpoint — the in-process `models.qwen` load never runs in this
# container (it requires SAGEMAKER_RANKER_ENDPOINT to be set, which the deployed
# task always provides). That keeps the image small and free of GPU/ML deps.
#
# Entry point: `python -m harvester.orchestrate` (a one-shot; EventBridge runs it
# on a schedule). Pass flags via the task definition command override.

# ---- builder: resolve + install the runtime deps with uv into a venv ----
FROM python:3.11-slim AS builder

# uv for fast, lockfile-faithful installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv

WORKDIR /app

# Only the metadata needed to resolve deps, so this layer caches unless deps change.
COPY pyproject.toml uv.lock README.md ./

# Build the venv with ONLY the `runtime` dependency-group — `--only-group`
# isolates it from the base deps, so torch/datasets/gradio/etc. never land in the
# image (~129 packages vs. ~265 for the full project).
# Export to a file first (RUN uses /bin/sh, which has no <(...) process
# substitution), then install from it.
RUN uv export --no-hashes --no-emit-project --only-group runtime --frozen \
      > /tmp/runtime-requirements.txt \
 && uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python -r /tmp/runtime-requirements.txt

# ---- final: slim runtime image ----
FROM python:3.11-slim AS runtime

# Non-root user — the task holds AWS credentials; don't run as root.
RUN groupadd --system app && useradd --system --gid app --home /app app

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# The resolved venv from the builder.
COPY --from=builder /opt/venv /opt/venv

# Only the source the runtime path imports: the agent graph, the harvester, and
# the prebuilt read-only RAG corpus. (models/ is intentionally omitted — its only
# runtime use is the in-process ranker, which the SageMaker path replaces.)
COPY agents/   ./agents/
COPY harvester/ ./harvester/
COPY .chroma/  ./.chroma/

USER app

# One-shot. Override args (e.g. --dry-run, --platforms) via the task command.
ENTRYPOINT ["python", "-m", "harvester.orchestrate"]
