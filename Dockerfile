# Multi-stage production image for the DCM syndicate engine.
# Stage 1 builds the wheel; stage 2 ships an upgraded slim runtime,
# patched against known OS CVEs and running as non-root.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY dcm_engine ./dcm_engine

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir build \
    && python -m build --wheel --outdir /wheels

FROM python:3.11-slim

# Patch OS packages and strip build tooling: the slim base ships stale
# point releases and vendored build tools (wheel, jaraco.context) that
# fail the Trivy HIGH/CRITICAL gate. None are needed at runtime.
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip uninstall --yes pip setuptools wheel

WORKDIR /app
COPY --from=builder /wheels /app/wheels

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run non-root (same uid as the distroless nonroot image).
USER 65532:65532

CMD ["python", "-c", "print('dcm-engine image: run a module entrypoint')"]
