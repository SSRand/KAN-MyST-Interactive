# Build from the repo root:
#   docker build -f app/Dockerfile -t kan-dashboard .
# This keeps the pyproject.toml + uv.lock at the root and copies the Dash
# source from app/.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/work/.venv \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PATH="/work/.venv/bin:${PATH}"

# uv ships an official, statically linked binary. Pulling it from a labelled
# image gives reproducible builds and bypasses pip-bootstrap.
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

WORKDIR /work

# Install deps in their own layer so app code changes don't bust the cache.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# App code (and only the app code — see .dockerignore at the build context root)
COPY app /work/app

WORKDIR /work/app

EXPOSE 8050

# 1 worker is enough: KAN training is single-threaded CPU-bound, so adding
# workers only trades RAM for no extra throughput.
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "1", "--timeout", "120"]
