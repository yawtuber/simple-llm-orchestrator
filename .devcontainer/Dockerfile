FROM python:3.13-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*