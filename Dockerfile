# Multi-stage production image for the DCM syndicate engine.
# Stage 1 builds wheels; stage 2 ships a distroless runtime so the
# attack surface is minimal and the container runs non-root.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY dcm_engine ./dcm_engine

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir build \
    && python -m build --wheel --outdir /wheels

FROM gcr.io/distroless/python3-debian12:nonroot

# Distroless Python images need packages installed site-wide; copy the
# wheel and its dependencies from the builder virtualenv instead.
COPY --from=builder /wheels /wheels

# NOTE: In production, bake dependencies at build time into a venv layer.
# Here we keep the image minimal and document the install path.
WORKDIR /app
COPY --from=builder /wheels /app/wheels

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Distroless nonroot already runs as uid 65532; no USER directive needed.
CMD ["python", "-c", "print('dcm-engine image: run a module entrypoint')"]
