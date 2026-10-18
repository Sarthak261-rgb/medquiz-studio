# Serves the API and the review screen over the committed runs and reviews.
# The generation/check pipeline needs a GPU, torch and Ollama, and runs on the host.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY src ./src
COPY review_app ./review_app
COPY configs ./configs
COPY prompts ./prompts
COPY data ./data
COPY runs ./runs
RUN uv sync --frozen && mkdir -p /app/var

ENV PATH="/app/.venv/bin:$PATH" \
    DATABASE_URL=sqlite:////app/var/medquiz.db
EXPOSE 8000 8501
HEALTHCHECK --interval=10s --timeout=3s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "medquiz db rebuild && uvicorn medquiz.api:app --host 0.0.0.0 --port 8000"]
