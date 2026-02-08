# AI Agent (RAG + Ticketing)

Intelligent Streamlit assistant for University that combines OpenAI Responses API, retrieval-augmented generation (RAG) over a Qdrant vector store, and BOS ticket drafting. Users can chat, upload documents or images, and ingest new materials into Qdrant for grounded answers.

---

## Features
- Conversational UI with chat history, image and document uploads (Streamlit chat).
- Grounded answers via RAG over Qdrant collections (`Documents`, `Knowledge`, `Processed`, `Tickets`).
- One-click ingestion pipeline with hashing-based deduplication, chunking, and OCR fallback for PDFs.
- Automatic category classification and ticket scaffolding for BOS (Biuro Obsługi Studenta).
- Configurable system prompt, categories, priorities, and model names via JSON in `settings/`.
- Sidebar observability for Qdrant collections and import progress.

## Quickstart
1) **Python environment** (3.10+ recommended):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) **System packages** (for PDF/OCR ingestion):
- Poppler utilities (`pdftoppm`, `pdftocairo`) for `pdf2image`.
- Tesseract OCR with Polish language data (`tesseract`, `tesseract-ocr-pol`).

3) **Environment variables** (via `.env` or Streamlit secrets):
```bash
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-5.1
EMBEDING_MODEL_NAME=text-embedding-3-large
QDRANT_URL=https://<qdrant-host>
QDRANT_API_KEY=<qdrant-key>
```

4) **Run the app**:
```bash
streamlit run app.py
```

## How it works
- **UI bootstrap**: [app.py](app.py) loads custom CSS, reads `settings/*.json`, and ensures Qdrant collections exist.
- **Conversation flow**: user text + optional images become input parts, cached in `st.session_state`. Messages render with Streamlit chat components.
- **RAG step**: `call_responses_api()` embeds the user prompt, queries Qdrant (`top_k=5`), and prepends retrieved chunks to the system prompt before calling `client.responses.create()`.
- **Responses API**: Uses `MODEL_NAME` (default GPT-5 class) with continuity via `previous_response_id` to preserve context across turns.
- **Uploads**: Files go to `documents/uploads/`; text files <1k chars are echoed, longer docs are embedded and upserted into Qdrant, then copied to `documents/processed/`.
- **Ticketing**: `Ticket` dataclass stores BOS ticket fields; `assign_category_and_priority()` asks the LLM to pick values from `settings/categories.json` and `settings/priorities.json`.

## Key components
- Chat & RAG orchestration: [tools/api_utils.py](tools/api_utils.py)
	- `build_input_parts()`: normalizes text and images into Responses API format.
	- `call_responses_api()`: embeds user text, fetches context from Qdrant, calls OpenAI Responses API.
	- `get_text_output()`: extracts assistant text.
	- `embed_text()` / `embed_image()`: helper embedding calls.
- Qdrant helpers: [tools/qdrant_utils.py](tools/qdrant_utils.py)
	- `ensure_collections_exist()`: idempotent collection creation.
	- `import_and_index_documents_qdrant()`: dedup by SHA256, chunk, embed, upsert with progress UI; OCR fallback for scanned PDFs.
	- `not_imported_files()`: counts unseen files by hash.
	- `extract_text_from_pdf()`: text-first, OCR-second extraction.
- Ticketing: [tools/ticket.py](tools/ticket.py)
	- `Ticket` dataclass with status, attachments, timestamps.
	- `assign_category_and_priority()`: LLM-based classification from configured lists.
- Prompt & formatting utilities: [tools/llm_utils.py](tools/llm_utils.py)
	- `build_system_prompt()`: flattens `settings/llm.json` into a single system string.
	- `determine_category_llm()`: classifies documents during ingestion.
- File moves: [tools/file_utils.py](tools/file_utils.py) for upload persistence and processed copies.

## Configuration
- `settings/llm.json`: persona, guardrails, and Markdown formatting rules for the system prompt.
- `settings/models.json`: reference list of available backends/embedders.
- `settings/categories.json`: BOS categories surfaced in chat and ingestion.
- `settings/priorities.json`: allowed ticket priorities.
- `settings/qdrant.json`: declarative list of collections (runtime uses env vars for URL/key).

## Data and directories
- `documents/knowledge/`: primary corpus to ingest (txt, md, pdf, docx, images).
- `documents/uploads/`: per-session uploads saved from the chat UI.
- `documents/processed/`: copied uploads after ingestion.
- `documents/logs/`: placeholder for runtime logs.
- `static/css/main.css`: Streamlit theming override.

## Ingestion workflow
1) Sidebar → **Import Documents** triggers `import_and_index_documents_qdrant()`.
2) Each file hashed (SHA256) to skip duplicates already in Qdrant.
3) Text is chunked (800 chars, 200 overlap) and embedded with `EMBEDING_MODEL_NAME`.
4) Payload stored with source path, category, chunk offsets, upload timestamp.
5) Images are embedded via filename proxy embedding for discoverability.

Programmatic example:
```python
from qdrant_client import QdrantClient
from openai import OpenAI
from tools.qdrant_utils import import_and_index_documents_qdrant

client = OpenAI()
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

import_and_index_documents_qdrant(
		qdrant_client=qdrant,
		client=client,
		embedding_model_name="text-embedding-3-large",
		knowledge_dir="documents/knowledge",
		model_name="gpt-5.1",
		settings_dir="settings",
)
```

## Chat + RAG example (standalone)
```python
from openai import OpenAI
from tools.api_utils import build_input_parts, call_responses_api, get_text_output
from qdrant_client import QdrantClient

client = OpenAI()
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

parts = build_input_parts("Jak uzyskać zaświadczenie o studiowaniu?", images=[])

resp = call_responses_api(
		system_prompt="You are a helpful university assistant.",
		client=client,
		model_name="gpt-5.1",
		embedding_model_name="text-embedding-3-large",
		parts=parts,
		qdrant_client=qdrant,
		qdrant_collection="Documents",
		top_k=5,
)

print(get_text_output(resp))
```

## Ticket drafting example
```python
from openai import OpenAI
from tools.ticket import Ticket, assign_category_and_priority

client = OpenAI()
context = "Potrzebuję zaświadczenia o studiowaniu do przedstawienia w urzędzie pracy."

selection = assign_category_and_priority(client, model_name="gpt-5.1", conversation_context=context)

ticket = Ticket(
		title="Prośba o zaświadczenie",
		description=context,
		created_by="student@uni.edu",
		category=selection["category"],
		priority=selection["priority"],
)

print(ticket.as_dict())
```

## Security and secrets
- Do not hardcode keys; prefer `.env` or Streamlit secrets.
- Uploaded files are stored locally; scrub if handling PII.
- Qdrant connectivity is authenticated via `QDRANT_API_KEY`.

## Troubleshooting
- Missing OCR/text from PDFs: install Poppler and Tesseract with Polish language data.
- Empty answers: verify `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and that collections contain vectors.
- Slow ingestion: adjust `batch_size` in `import_and_index_documents_qdrant()`; reduce chunk size if needed.

## Running in production
- Front behind HTTPS and secure secrets management.
- Use persistent storage for `documents/` and Qdrant collections.
- Enable Streamlit auth or reverse-proxy auth if exposed publicly.

---

Happy building! If you extend the agent, keep system prompt, categories, and ingestion schema aligned to ensure consistent responses.
