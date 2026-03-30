# Cat Health RAG

An evidence-based Retrieval-Augmented Generation (RAG) system for answering cat health questions using vetted veterinary sources.

This project retrieves relevant medical information from curated sources, generates grounded answers using an LLM, includes citations, and flags potentially urgent symptoms.

---

## Project Structure

```txt
cat-health-rag/
├── .venv/
├── backend/
│   ├── data/
│   │   ├── raw/              # raw HTML + metadata
│   │   ├── processed/        # cleaned text + metadata
│   │   ├── chroma/           # vector database
│   │   └── sources.yaml      # source definitions
│   ├── scripts/
│   │   ├── fetch_sources.py  # download raw HTML
│   │   └── parse_html.py     # clean + extract text
│   └── src/
│       └── petmed_rag/
│           ├── api/          # FastAPI app
│           ├── cli/          # CLI interface
│           ├── generation/   # answer generation + prompts
│           ├── retrieval/    # vector search logic
│           ├── safety/       # urgency detection
│           ├── ingestion/    # chunking + ingestion
│           ├── embeddings.py
│           └── config.py
```

---

## Setup

From the project root, activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Offline Ingestion (Required First)

Before asking any questions, you must build the knowledge base.

From `backend/`:

```powershell
python .\scripts\fetch_sources.py
python .\scripts\parse_html.py
python -m petmed_rag.cli.main ingest
```

### What this does

#### `fetch_sources.py`

- Downloads raw HTML from curated veterinary sources

#### `parse_html.py`

- Extracts clean article text
- Removes HTML noise such as scripts, navigation, and other irrelevant page elements

#### `ingest`

- Chunks text
- Generates embeddings
- Stores vectors in ChromaDB

---

## Online Query

After ingestion, you can query the system in two ways:

### Option 1: CLI

From `backend/`:

```powershell
python -m petmed_rag.cli.main ask "Why is my cat vomiting?"
```

### Option 2: API (FastAPI)

From `backend/`:

```powershell
uvicorn petmed_rag.api.main:app --reload
```

Then open:

- Health check: `http://127.0.0.1:8000/health`
- Docs: `http://127.0.0.1:8000/docs`

---

## Sample Request

```json
{
  "question": "Why is my cat vomiting?"
}
```

## Sample Response

```json
{
  "answer": "Your cat may be throwing up for various reasons, ranging from minor issues like hairballs or eating too quickly to more serious conditions such as blockages, infections, or organ disease. Occasional vomiting may not be urgent, but repeated vomiting, blood, lethargy, or refusal to eat are signs that require veterinary attention.",
  "citations": [
    {
      "id": "gsvc-vomiting_c2",
      "title": "Cat Throwing Up: Is It a Veterinary Emergency?",
      "url": "https://gsvs.org/blog/cat-throwing-up-emergency/"
    },
    {
      "id": "winthrop-vet-vomiting_c1",
      "title": "Why Is My Cat Throwing Up (Again)? A Fur-Real Guide to Feline Vomit",
      "url": "https://winthropvet.net/news/why-is-my-cat-throwing-up-again-a-fur-real-guide-to-feline-vomit"
    }
  ],
  "safety_flag": "urgent",
  "disclaimer": "This may be urgent. Please contact a veterinarian or emergency vet as soon as possible."
}
```

---

## How It Works

1. The user question is embedded.
2. The most relevant chunks are retrieved from ChromaDB.
3. Retrieved context is passed to the LLM.
4. The model generates a grounded answer.
5. The system returns:
   - `answer`
   - `citations`
   - `safety_flag`

---

## Notes

- This system is for informational purposes only and is not a substitute for veterinary care.
- Data quality, including clean parsing and metadata, is critical for strong retrieval performance.
- The ingestion step must be run before querying.

---

## Future Improvements

- React + SCSS chat frontend
- More veterinary sources
- Improved ranking / reranking
- Better safety detection
- Incremental indexing

---

## Author

Angelina Rodriguez
