FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system -e ".[dev]"

COPY app/ ./app/
COPY tests/ ./tests/

RUN mkdir -p /app/data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
