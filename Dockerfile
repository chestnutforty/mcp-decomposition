FROM python:3.12-slim

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ARG GITHUB_TOKEN
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

COPY . .
RUN rm -rf .venv uv.lock

RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# Default port
ENV PORT=8000
EXPOSE ${PORT}

# Run the MCP server
# Override ENTRY_SCRIPT env var to use a different entry point
ENV ENTRY_SCRIPT=server.py
CMD uv run fastmcp run -t sse ${ENTRY_SCRIPT}:mcp --port ${PORT} --host 0.0.0.0
