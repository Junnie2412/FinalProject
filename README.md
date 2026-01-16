# AI-Powered Software Testing Automation App (AI Test Agent)

A prototype platform for AI-assisted software testing automation.  
This project focuses on two core building blocks:

1) **RAG Knowledge Base from Project Documents (PDF → Chroma VectorDB → Ollama LLM)**  
2) **Controlled Web Search & Scraping (Search → Playwright Rendered HTML → Text Extraction → Optional Indexing)**

The result is a reusable pipeline for building a project-aware AI testing assistant, and a foundation for next steps such as test-case generation and UI automation.

---

## Key Features (Implemented)

### RAG + VectorDB (PDF/Text → Chroma)
- Load PDFs using `PyPDFLoader` (LangChain).
- Chunking using `RecursiveCharacterTextSplitter` (configurable `chunk_size`, `chunk_overlap`).
- Embeddings via **Ollama embeddings model** (default: `embeddinggemma`).
- Storage via **Chroma VectorDB** persisted to `data/chroma/`.
- Query via **Ollama chat model** (default: `qwen3:1.7b`) with retrieved context.
- Optional extraction/dump of PDF text for verification.
- Ability to index **plain text files** into Chroma (used for web text indexing).
- Retrieval supports metadata filtering (`where`) and is compatible with Chroma versions that use `filter=...` vs `where=...`.

### Controlled Web Search & Scraping
- Keyword URL search:
  - Attempts Google search via `googlesearch-python` (if accessible).
  - **Fallback:** direct Viblo search (Playwright-based) when Google is blocked or empty.
  - Optional domain allowlist filtering.
- Web scraping via **Playwright**:
  - Loads fully rendered HTML (`page.content()`).
  - Extracts text with **lxml**.
  - Removes `script/style/noscript`.
  - Optional **CSS selector** (`content_selector`) to extract main content only.
  - Domain allowlist enforcement.
  - Configurable timeout and headless/non-headless mode.
- Save scraped artifacts:
  - `outputs/page.html`
  - `outputs/page.txt`

### Integrated Pipeline (One Command)
- Single command pipeline:
  - **PDF mode**: ingest PDFs → query
  - **URL mode**: scrape → save → index text into Chroma with metadata → query with URL-only filtering
- In URL mode, retrieval is filtered by metadata:
  - `{"$and": [{"source_type": "web"}, {"url": "<target_url>"}]}`
  - This prevents mixing content across different URLs.

### Debug Outputs (Implemented)
- Save debug artifacts for inspection:
  - `outputs/last_question.txt`
  - `outputs/last_answer.txt`
  - `outputs/last_context.txt`
  - `outputs/last_hits.json`

### MCP Tool Server (Implemented)
- Exposes the main capabilities as MCP tools:
  - `atp_web_search`
  - `atp_web_scrape`
  - `atp_web_index`
  - `atp_rag_ingest`
  - `atp_rag_query`
  - `atp_run`
- Supports `streamable-http` (recommended) and `stdio` transports.

---

## Project Structure

```
FinalProject/
  docs/                  # Put PDFs here (example input)
  data/chroma/           # Chroma persisted vector DB
  outputs/               # Scraped files + debug outputs
  src/atp/
    cli.py               # Typer CLI entrypoint (console script: atp)
    rag/rag_core.py       # PDF ingest, text indexing, retrieve, answer
    web/search.py         # URL search (Google + Viblo fallback)
    web/scrape.py         # Playwright scrape + lxml extraction
    web/index.py          # Index scraped text into Chroma
    mcp_server.py         # MCP tool server
  requirements.txt
  pyproject.toml
```

---

## Requirements

- Python **3.11+**
- **Ollama** installed and running locally
- Playwright browser installed (Chromium)

---

## Setup

### 1) Create environment & install dependencies

Using `pip`:
```bash
pip install -r requirements.txt
```

Or using `uv` (if you prefer):
```bash
uv sync
```

### 2) Install Playwright browser
```bash
playwright install chromium
```

### 3) Prepare Ollama models
Pull the default models (or choose your own):
```bash
ollama pull embeddinggemma
ollama pull qwen3:1.7b
```

Make sure Ollama is running.

---

## CLI Usage (Typer)

The project installs a console script named: **`atp`**

Show help:
```bash
atp --help
```

### A) PDF → Chroma ingest

Ingest all PDFs inside `docs/`:
```bash
atp rag-ingest --docs-dir docs --chroma-dir data/chroma --embed-model embeddinggemma
```

Optionally dump extracted PDF text to verify loader output:
```bash
atp rag-ingest --docs-dir docs --dump-text --out-dir outputs
```

### B) RAG query (from the current Chroma DB)

```bash
atp rag-query "What is the scope of this project?" --chroma-dir data/chroma --chat-model qwen3:1.7b
```

Save debug artifacts (enabled by default):
```bash
atp rag-query "..." --save-debug true --out-dir outputs
```

### C) Web search (URLs)

Search for URLs:
```bash
atp web-search "RAG Chroma Ollama tutorial" --limit 5
```

Allow only a specific domain (can repeat the option):
```bash
atp web-search "Playwright scraping" --allowed-domain viblo.asia --limit 5
```

**Note:** If Google search is blocked or returns nothing, the implementation falls back to Playwright-based Viblo search.

### D) Web scrape (HTML + extracted text)

Scrape a URL and save results to `outputs/`:
```bash
atp web-scrape "https://example.com" --out-dir outputs
```

Use a content selector (extract main content only):
```bash
atp web-scrape "https://example.com/article" --content-selector "article"
```

Enforce domain allowlist:
```bash
atp web-scrape "https://viblo.asia/p/..." --allowed-domain viblo.asia
```

Disable headless (open browser window):
```bash
atp web-scrape "https://example.com" --no-headless
```

### E) Index scraped web text into Chroma

After scraping, index `outputs/page.txt` into Chroma with URL metadata:
```bash
atp web-index --text-path outputs/page.txt --url "https://example.com/article" --chroma-dir data/chroma
```

### F) One-command pipeline (PDF mode or URL mode)

#### PDF mode
Ingest PDFs then query:
```bash
atp run "Summarize the key points" --pdf-dir docs --chroma-dir data/chroma
```

#### URL mode
Scrape → index → query (retrieval filtered to that URL only):
```bash
atp run "What does this page say about X?"   --url "https://example.com/article"   --allowed-domain example.com   --content-selector "article"   --chroma-dir data/chroma   --out-dir outputs
```

---

## MCP Tool Server

This project includes an MCP server exposing the core functions as tools.

### Run (recommended transport: streamable-http)
```bash
python -m atp.mcp_server --transport streamable-http
```

### Run (stdio transport)
```bash
python -m atp.mcp_server --transport stdio
```

**Important for `stdio`:**
- Do not print to stdout (it can corrupt JSON-RPC).
- This server uses logging to stderr to avoid that issue.

### Available MCP Tools
- `atp_web_search(query, limit=5, allowed_domain=None)`
- `atp_web_scrape(url, allowed_domain=None, content_selector=None, headless=True, timeout_ms=30000, out_dir="outputs")`
- `atp_web_index(url, text_path="outputs/page.txt", chroma_dir="data/chroma", embed_model="embeddinggemma")`
- `atp_rag_ingest(docs_dir="docs", chroma_dir="data/chroma", embed_model="embeddinggemma")`
- `atp_rag_query(question, chroma_dir="data/chroma", embed_model="embeddinggemma", chat_model="qwen3:1.7b", top_k=4, url=None)`
- `atp_run(question, url=None, pdf_dir=None, allowed_domain=None, content_selector=None, chroma_dir="data/chroma", embed_model="embeddinggemma", chat_model="qwen3:1.7b", top_k=4, out_dir="outputs")`

---

## Notes / Design Details

### 1) Metadata filtering for web content
When indexing web text, the system attaches metadata:
- `source_type = "web"`
- `url = "<page_url>"`

When querying, it can filter retrieval to exactly one URL:
```json
{
  "$and": [
    {"source_type": "web"},
    {"url": "https://example.com/article"}
  ]
}
```

This avoids mixing multiple web sources in one answer.

### 2) Chroma API compatibility
Some Chroma/LangChain versions use `filter=` while others use `where=` for metadata filters.  
The implementation attempts `filter=` first and falls back to `where=` to remain compatible.

### 3) Guardrail behavior
The current RAG prompt instructs the model to answer only using the retrieved context, otherwise return a fixed “not enough information” style response.  
(You can customize this prompt easily inside `src/atp/rag/rag_core.py`.)

---

## Roadmap (Suggested Next Steps)
- Test-case generation from requirements and specs stored in the RAG knowledge base.
- UI automation agent (Playwright actions) driven by LLM plans.
- Evidence collection: screenshots, DOM snapshots, logs, and trace artifacts.
- Evaluation harness: compare expected vs actual, and generate test reports.

---

## License
Prototype / internal research use. Add a license if you plan to publish.
