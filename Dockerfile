# DocIntel container image. Built for Hugging Face Spaces (Docker SDK, port 7860) but runs anywhere:
#   docker build -t docintel .
#   docker run -p 7860:7860 docintel
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    # The public demo ships with the LLM off; add GEMINI_API_KEY/GROQ_API_KEY as secrets and set
    # LLM_PROVIDER to turn it on.
    LLM_PROVIDER=none

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
USER user
ENV PATH=/home/user/.local/bin:$PATH

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Bake the embedding model into the image so the first upload isn't a 90 MB download
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user main.py .
COPY --chown=user api ./api
COPY --chown=user src ./src
COPY --chown=user web ./web
COPY --chown=user models ./models
COPY --chown=user sample_docs ./sample_docs

EXPOSE 7860
# One worker: documents and the vector index live in process memory.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
