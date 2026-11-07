# API image: FastAPI + LangGraph pipeline + local embedding model.
# Build:  docker build -t hr-assistant-api .     (CI adds --build-arg INSTALL_DEV=true)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBEDDING_CACHE_DIR=/app/models
WORKDIR /app

# 1. Dependencies (rebuilt only when pyproject.toml changes)
COPY pyproject.toml .
ARG INSTALL_DEV=false
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml', 'rb')); print('\n'.join(d['project']['dependencies']))" > requirements.txt \
 && pip install -r requirements.txt \
 && if [ "$INSTALL_DEV" = "true" ]; then \
      python -c "import tomllib; d = tomllib.load(open('pyproject.toml', 'rb')); print('\n'.join(d['dependency-groups']['dev']))" > requirements-dev.txt \
      && pip install -r requirements-dev.txt; \
    fi

# 2. Bake the embedding model into the image, so containers start without downloading it
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models')"

# 3. Code, policies and evaluation set
COPY hr_assistant ./hr_assistant
COPY scripts ./scripts
COPY tests ./tests
COPY data/policies ./data/policies
COPY data/eval ./data/eval

# 4. Non-root user; data/runtime is the volume (index + telemetry)
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p data/runtime eval/results \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "hr_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
