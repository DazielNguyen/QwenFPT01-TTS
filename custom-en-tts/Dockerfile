# Multi-stage Dockerfile for qwenfpt-tts demo.
# Checkpoints downloaded from HuggingFace Hub at container startup — NOT baked in.
# Target image size: < 5 GB (excluding checkpoint downloads).
# Exposes Gradio demo on port 7860.
#
# Build: docker build -t qwenfpt-tts .
# Run:   docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN qwenfpt-tts
#
# Secrets: pass HF_TOKEN as env var — never hardcode in Dockerfile or image layers.

# ---- Build stage ----
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt && \
    pip install --no-cache-dir --user gradio>=4.0.0

# ---- Runtime stage ----
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from build stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code (not data or checkpoints)
COPY src/ ./src/
COPY demo/ ./demo/
COPY configs/ ./configs/
COPY setup.py .

# Port for Gradio
EXPOSE 7860

# Checkpoints downloaded at startup via HF_TOKEN env var
# HF_TOKEN must be provided at docker run time
ENV HF_TOKEN=""

CMD ["python", "demo/gradio_app.py"]
