# DocIntel

Upload invoices, resumes, contracts or reports (PDF, PNG, JPG, DOCX) and DocIntel will read them, work out what
kind of document each one is, pull out the key fields, summarise them, and answer questions about them with
citations. It refuses to answer when the documents don't contain the information.

One FastAPI app serves both the JSON API and the web UI. An LLM (Gemini or Groq, free tiers) is optional: without
a key everything still works using rule-based extraction and extractive fallbacks.

## How it works

| Stage | What happens | Code |
|---|---|---|
| Ingestion | File type is detected from magic bytes. Digital PDFs use PyMuPDF; scans and images fall back to Tesseract OCR; DOCX is read directly. Text keeps its page numbers. | `src/ingestion.py`, `src/ocr.py` |
| Classification | TF-IDF + logistic regression over `invoice / resume / contract / report / other`. No LLM. | `src/classifier.py` |
| Extraction | Rule-based + spaCy baseline for invoices, resumes and contracts. If an LLM is configured, it extracts independently and the two results are compared field by field. | `src/extraction.py` |
| Summary | LLM summary with unsupported claims dropped, or a frequency-based extractive summary as fallback. | `src/summarize.py` |
| Q&A | Chunks are embedded (MiniLM) into a FAISS index. The top passages go to the LLM, which must cite them. Answers whose numbers or wording aren't supported by the cited text are refused. | `src/rag.py` |
| Safety | Lines that look like prompt injection are stripped before text reaches the LLM, and document text is wrapped in per-request random markers and treated as untrusted. | `src/extraction.py`, `tests/test_prompt_injection.py` |

Uploads are never kept on disk. Extracted text lives in memory and expires after `RETENTION_MINUTES`.

## Quick start

Requires Python 3.11 and [Tesseract](https://github.com/tesseract-ocr/tesseract) (needed only for scans and
images; on Windows set `TESSERACT_CMD` if it isn't on `PATH`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm

cp .env.example .env               # optional: add a Gemini or Groq key
uvicorn main:app
```

Open http://127.0.0.1:8000. The first upload downloads the embedding model (~90 MB).

The trained classifier is committed (`models/classifier.joblib`). To retrain it from scratch:

```bash
python -m training.generate_data
python -m training.train_classifier
```

## Configuration

Settings come from environment variables or `.env` (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq`, `gemini` or `none`. The LLM is used only if the matching key is set. |
| `GROQ_API_KEY` / `GEMINI_API_KEY` | empty | Free keys: [Groq](https://console.groq.com/keys), [Gemini](https://aistudio.google.com/apikey) |
| `LLM_RPM` | `14` | Client-side request limit, just under Gemini's free tier |
| `MAX_UPLOAD_MB` | `10` | Per-file upload limit |
| `RETENTION_MINUTES` | `60` | How long documents stay in memory |
| `RATE_LIMIT_UPLOAD` / `RATE_LIMIT_ASK` | `10/minute` / `20/minute` | Per-IP limits |
| `CORS_ORIGINS` | `*` | Restrict this if you deploy publicly |

## API

All routes are under `/api`; interactive docs are at `/docs`.

| Method and path | Purpose |
|---|---|
| `GET /health` | Liveness, and whether an LLM is configured (never the key) |
| `POST /upload` | Multipart `files` (up to 5). Reads, classifies and indexes each one. |
| `GET /documents` | List uploaded documents |
| `GET /documents/{id}` | Metadata and per-page text |
| `DELETE /documents/{id}` | Remove a document and its index entries |
| `POST /documents/{id}/extract` | Field extraction. Body: `{"use_llm": true}` |
| `POST /documents/{id}/summary` | Summary and key points |
| `POST /ask` | `{"question": "...", "doc_ids": ["..."]}` → answer with citations, or a refusal |

```bash
curl -F files=@sample_docs/sample_invoice.pdf http://127.0.0.1:8000/api/upload
curl -X POST localhost:8000/api/ask -H 'Content-Type: application/json' \
     -d '{"question": "What is the total amount due?"}'
```

## Tests and evaluation

```bash
ruff check .
pytest
```

CI runs both on every push and pull request.

`python -m evaluation.run_eval` scores the whole pipeline on 75 held-out documents (`evaluation/test_docs/`,
15 per type, spread across PDF, DOCX, PNG and JPG). It runs offline and never calls an LLM. Add `--llm` to
also evaluate LLM extraction and generated answers (needs a key; takes about 20 minutes at the free-tier rate
limit). Results go to `data/eval_report.json`.

Results (75 documents; the LLM column is `gemini-flash-lite-latest`):

| Stage | Offline (no LLM) | With LLM |
|---|---|---|
| Classification accuracy | 100% | 100% (no LLM involved) |
| Field extraction F1 | 0.902 (rule-based) | 0.985 LLM alone, **0.992** merged |
| Q&A: right passage retrieved | 99.3% | 99.3% |
| Q&A: answerable questions answered correctly | 78.8% | **98.5%** |
| Q&A: unanswerable questions refused | 29.0% | **98.6%** |

Read these with care:

- The classifier is trained on synthetic documents and tested on documents from the same generator, so the 100%
  almost certainly overstates accuracy on real-world files. The extraction and Q&A documents are synthetic too.
- Without an LLM, the only thing that can refuse a question is the retrieval-score cutoff, so the low offline
  refusal rate is expected: the extractive fallback returns the closest passage instead.
- In the LLM run, 1 of 45 extractions had an unusable model reply (kept the rule-based result) and 2 of 206
  answers hit the free-tier rate limit and fell back to a passage; both are counted in the scores above.
- Rule-based extraction is weakest on contract parties (F1 0.70), invoice vendor (0.73) and currency (0.80);
  the LLM cross-check mostly fixes these.

## Deploying

A `Dockerfile` is included (Tesseract, the spaCy model and the embedding model are baked in; the app listens on
port 7860 with a single worker). It ships with `LLM_PROVIDER=none`, so a public demo can't spend anyone's API
quota. To run it locally: `docker build -t docintel . && docker run -p 7860:7860 docintel`.

To publish on [Hugging Face Spaces](https://huggingface.co/new-space) (free, 16 GB RAM):

1. Create a Space with the **Docker** SDK and **Public** visibility, plus a write token from your HF settings.
2. `HF_TOKEN=hf_xxx deploy/push_space.sh <username>/<space-name>` pushes the committed `HEAD` (with the header
   Spaces needs added to the README; this repo's README is unchanged).
3. Optional: to turn the AI on, add `LLM_PROVIDER` and the matching `GEMINI_API_KEY` / `GROQ_API_KEY` in the
   Space's *Settings → Variables and secrets*. Anyone can then use that quota, so prefer a separate key.

Behind a reverse proxy the per-IP rate limits use the address the proxy reports in `X-Forwarded-For`.

## Project layout

```
api/          FastAPI routes and the in-memory document store
src/          ingestion, OCR, classifier, extraction, RAG, summaries, LLM providers
web/          static UI (plain HTML/CSS/JS, no build step)
training/     synthetic data generator and classifier training
evaluation/   run_eval.py and scoring metrics, plus the held-out test documents
sample_docs/  a few documents to try
tests/        pytest suite
```

## Limitations

- State is in memory and per process: restarting the server drops all documents, and it won't work behind
  multiple workers.
- There is no authentication. If you expose it publicly, restrict `CORS_ORIGINS` and keep the upload limits.
- Free LLM tiers are rate-limited; when the model is unavailable, answers fall back to the most relevant passage
  and are labelled as such.
- English documents only.
