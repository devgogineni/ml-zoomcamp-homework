FROM python:3.12-slim

# xgboost's linux wheels need the OpenMP runtime, which the slim image lacks
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# install dependencies first so this layer is cached until pyproject.toml/uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# run commands from the project virtual environment
ENV PATH="/app/.venv/bin:$PATH"

COPY training.py ./
COPY data/solar_energy_production_data_for_2025.csv ./data/

# the trained model is written to /app/model; mount a volume there to keep it
VOLUME /app/model

CMD ["python", "training.py"]
