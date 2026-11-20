# CPU-only image for Aegis. Works on Hugging Face Spaces (Docker SDK, port 7860).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    AEGIS_BACKEND=local \
    PORT=7860

# Non-root user (Hugging Face Spaces also runs containers as uid 1000)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt constraints.txt ./
# CPU-only torch first (the default PyPI wheel drags in ~2 GB of CUDA libraries), then the rest, pinned.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu -c constraints.txt \
 && pip install -r requirements.txt -c constraints.txt

COPY --chown=user:user aegis ./aegis
COPY --chown=user:user app ./app
COPY --chown=user:user corpus ./corpus
COPY --chown=user:user eval/threshold.json ./eval/threshold.json
COPY --chown=user:user scripts/download_models.py ./scripts/download_models.py

USER user
# Bake the model weights into the image so the container never downloads at start-up.
RUN python -m scripts.download_models

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"7860\")}/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
