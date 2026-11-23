# Slim CPU image: ONNX models via fastembed (no torch). Honours $PORT (Render sets it).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    AEGIS_MODEL_DIR=/home/user/models \
    AEGIS_BACKEND=local \
    PORT=7860

# Non-root user
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

COPY --chown=user:user aegis ./aegis
COPY --chown=user:user app ./app
COPY --chown=user:user docs ./docs
COPY --chown=user:user corpus ./corpus
COPY --chown=user:user eval/threshold.json ./eval/threshold.json
COPY --chown=user:user scripts/download_models.py ./scripts/download_models.py

USER user
# Bake the two ONNX models into the image so the container never downloads at start-up.
RUN python -m scripts.download_models
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"7860\")}/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
