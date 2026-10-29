# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS build
WORKDIR /src
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install ".[anthropic,redis,mcp]"

FROM python:3.12-slim
# non-root runtime user; the only writable path is the data volume
RUN useradd --system --uid 10001 --home-dir /data tap && mkdir -p /data && chown tap /data
COPY --from=build /install /usr/local
COPY data /app/data
WORKDIR /app
USER tap
ENV TAP_DATA_DIR=/data PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"
CMD ["tap", "serve", "--host", "0.0.0.0", "--port", "8000"]
