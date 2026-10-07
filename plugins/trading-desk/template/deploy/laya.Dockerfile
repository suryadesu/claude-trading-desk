# The decision model, as a sidecar: Laya (huggingface.co/convaiinnovations/laya,
# Apache-2.0) behind `laya-serve`, which speaks the /v1/systemone protocol the
# bots' ModelDecider already uses. It runs in its own image because laya needs
# Python >= 3.10 and torch, and the bots should not carry either.
#
# CPU-only torch. The download.pytorch.org CPU index has both amd64 and arm64
# wheels, so this builds natively on Oracle's Ampere A1 with no emulation.
# Pin LAYA_VERSION once you have a version you have calibrated against: a new
# checkpoint is a new model, and the cached decisions no longer describe it.
FROM python:3.11-slim-bookworm

ARG LAYA_VERSION=""
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models LAYA_HOST=0.0.0.0 LAYA_PORT=8000

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir "laya[serve]${LAYA_VERSION:+==$LAYA_VERSION}"

RUN useradd --uid 10001 --create-home laya && mkdir -p /models && chown laya /models
USER laya
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["laya-serve"]
