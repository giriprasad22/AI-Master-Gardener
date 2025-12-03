# AI Gardener

A local-first AI assistant for sustainable agriculture that blends classic machine learning (crop and disease vision models) with retrieval augmented generation, companion planting graphs, and yield analytics. Everything runs on Windows with Python 3.11, TensorFlow, and a single local large language model (LLM) served by Ollama, so you can operate completely offline after the initial setup.

```
User ──► Flask Web UI ──► Agent Planner (Ollama gemma3) ──► Tools & Models
                                   │
                                   ├─► Supabase pgvector KB (optional)
                                   ├─► TensorFlow crop classifier
                                   ├─► TensorFlow disease classifier
                                   ├─► Companion planting graph analytics
                                   └─► Yield regression analytics
```

## Highlights

- Conversational agent that classifies intents (reply, search, detect crop, detect disease, companions, yield) and routes to the correct tool automatically.
- Local-only reasoning powered by `gemma3:latest` through Ollama (no external APIs required).
- Knowledge base retrieval via Supabase pgvector with automatic quality checks and fallback to LLM knowledge when necessary.
- Vision pipelines for crop identification (MobileNet) and disease diagnosis (custom CNN, 18 epochs, 38 classes).
- Companion planting insights backed by graph CSVs plus optional KB enrichment.
- Yield estimator that fits crop-specific linear models on demand from historical CSV data.
- Responsive frontend with modern UI, organic-first branding, image uploads, and multi-threaded chat history stored locally.

## Project Layout

```
.
├─ app/                  # Flask application, templates, static assets, API routes
├─ kb/                   # Supabase pgvector tooling (SQL, ingestion, search)
├─ crop_detection/       # Crop classifier models, notebooks, dataset assets
├─ crop_disease/         # Disease classifier models, training scripts, data helpers
├─ companion_plants/     # Companion/avoid CSV graphs and analysis script
├─ Yield_prediction/     # Yield dataset, notebook, and training artifacts
on notes, guides
├─ requirements.txt      # Consolidated project dependencies (Python 3.11)
└─ .venv311/             # Recommended local Python 3.11 virtual environment (not committed)
```

Key model artifacts expected by the app:

| Purpose | Path | Notes |
|---------|------|-------|
| Crop classifier | `crop_detection/best_crop_detection_model.keras` (or `crop_detection_model.keras`) | MobileNet-based 224x224 RGB model |
| Crop labels | `crop_detection/class_indices.pkl` | Directory-to-class mapping |
| Disease classifier | `crop_disease/trained_model_18epochs.h5` | TensorFlow 2.16 compatible CNN (128x128 RGB, raw 0-255) |
| Disease labels | `crop_disease/class_indices.pkl` | Generated during training |
| Yield data | `Yield_prediction/yield_df.csv` | Used to fit OLS regressions |
| KB docs | `kb/docs/*.txt|md|pdf` | Source files for Supabase ingestion |

## Technology Stack

- **Frontend:** HTML5, Vanilla JS, CSS (glassmorphism + responsive layout) served via Flask.
- **Backend:** Flask, Python 3.11, modular tool handlers in `app/app.py`.
- **LLM:** Ollama hosting `gemma3:latest` for planning, synthesis, and embeddings.
- **ML:** TensorFlow/Keras for image classifiers; scikit-learn/statsmodels-style regression for yield estimates.
- **KB:** Supabase Postgres with pgvector (`match_chunks` RPC) and local ingestion scripts using Ollama embeddings.
- **Data:** Companion planting CSV networks, PlantVillage disease dataset derivatives, crop detection datasets, rainfall/temperature/yield CSVs.

## Prerequisites

- Windows 10/11 machine.
- Python 3.11.x (tested with CPython 3.11; create a virtual environment such as `.venv311`).
- [Ollama](https://ollama.ai) desktop app (ensure the service is running).
- Pull the LLM and embedding models you plan to use, for example:
  ```powershell
  ollama pull gemma3:latest
  ollama pull nomic-embed-text
  ```
- (Optional) Supabase project with pgvector enabled for the knowledge base.
- (Optional) GPU acceleration for TensorFlow; CPU works but is slower.

## Initial Setup

From the repository root (`C:\Users\Giri prasad\Desktop\capstone-Project`):

```powershell
# 1. Create and activate the virtual environment (first-time only)
python -m venv .venv311
& .\.venv311\Scripts\Activate.ps1

# 2. Install shared dependencies for the entire project
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Optional: install or upgrade module-specific extras when needed
# (Only required if you plan to hack on those modules directly)
python -m pip install -r kb\requirements.txt   # Supabase tooling
python -m pip install tensorflow==2.16.*        # Ensures TF matches retrained models
```

To deactivate the environment later run `deactivate`.

### Environment Variables

The Flask app automatically loads `kb/.env` if present. Typical settings:

```
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_ANON_KEY=...
EMBEDDING_MODEL=gemma3:latest          # or your preferred embedding model
FAKE_EMBEDDINGS=1                      # optional, forces mock embeddings for quick demos
```

Set or clear variables in PowerShell before launching the app, for example:

```powershell
# Use mock embeddings for quick UI tests
$env:FAKE_EMBEDDINGS = "1"

# Clear the flag to use real embeddings after Supabase is configured
Remove-Item Env:FAKE_EMBEDDINGS
```

## Running the Web Assistant

```powershell
& .\.venv311\Scripts\Activate.ps1
python app\app.py
```

Visit http://127.0.0.1:8000 for the modern UI or http://127.0.0.1:8000/classic for the legacy layout retained for comparison. The app works in mock mode even without Supabase or TensorFlow; enable the optional services to unlock all tools.

## Knowledge Base (Optional but Recommended)

1. **Provision Supabase** and enable pgvector.
2. Run the SQL migrations:
   ```
   kb/sql/001_enable_pgvector.sql
   kb/sql/010_schema.sql
   ```
3. Copy `kb/.env.example` to `kb/.env` and fill in Supabase credentials and embedding model name.
4. Install KB dependencies and ingest documents:
   ```powershell
   & .\.venv311\Scripts\Activate.ps1
   python -m pip install -r kb\requirements.txt
   python kb\ingest.py
   ```
5. Test retrieval:
   ```powershell
   python kb\search.py "what grows well with tomatoes?"
   ```
6. Remove the `FAKE_EMBEDDINGS` flag and restart the Flask app to serve live KB-backed answers.

## Models and Training Pipelines

### Crop Classification (`/api/detect`)
- Default expectation: `crop_detection/best_crop_detection_model.keras` (MobileNet) and `crop_detection/class_indices.pkl`.
- Input preprocessing: resize to 224x224, normalize to [0, 1], batch dimension = 1.
- Training notebooks live in `crop_detection/` if you want to retrain on new data.

### Disease Classification (`/api/disease`)
- Primary model: `crop_disease/trained_model_18epochs.h5` (TensorFlow 2.16 compatible).
- Input preprocessing: resize to 128x128, retain raw 0-255 pixel values (no scaling), add batch dimension.
- Label mapping: `crop_disease/class_indices.pkl` (value = neuron index).
- Retraining script: `crop_disease/train_disease_model.py`
  ```powershell
  & .\.venv311\Scripts\Activate.ps1
  cd crop_disease
  python train_disease_model.py
  ```
  Adjust epochs, augmentation, or callbacks in the script to target higher accuracy. Outputs include `trained_model_XXepochs.h5`, `class_indices.pkl`, and training history JSON.

### Yield Prediction (`/api/yield`)
- Uses `Yield_prediction/yield_df.csv` to fit simple ordinary least squares models per crop at runtime.
- Accepts crop name, average temperature (°C), and rainfall (mm); returns tonnes per hectare plus qualitative advice.

### Companion Planting (`/api/companions`)
- Data sources: `companion_plants/help_network.csv`, `companion_plants/avoid_network.csv`.
- Graph queries identify helpers and antagonists and can cross-reference the KB for additional context.

## Using the Assistant

| Action | How to Trigger | What Happens |
|--------|----------------|--------------|
| General advice | Type any farming question | Agent searches KB, validates quality, and responds. Falls back to LLM knowledge if needed. |
| Crop detection | Upload plant photo + ask to identify | TensorFlow crop model predicts top-3 classes, agent enriches with growing tips. |
| Disease detection | Upload symptomatic plant photo + ask for diagnosis | Disease model predicts 38 diseases, agent returns symptoms, treatments, prevention, and organic remedies. |
| Companion planting | Ask for companions/avoid plants | Companion CSV graph returns helpers, antagonists, reasons, and disease mitigation hints. |
| Yield estimation | Provide crop, temperature, rainfall | On-demand regression predicts yield and suggests optimizations. |
| Search | Use Search tab or plan triggered by agent | Direct Supabase pgvector query with configurable top-k. |

Chat memories per thread are persisted as JSON under `app/memory/`, so follow-up questions automatically reuse context (e.g., “How do I treat it?” after a diagnosis).

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Modern UI (improved layout).
| GET | `/classic` | Legacy UI retained for comparison/testing.
| POST | `/api/chat` | Main agent entry point. Form fields: `thread_id`, `message`, optional `image`. Returns message text, citations, agent plan, and tool outputs. |
| POST | `/api/search` | KB search (`{"query": "...", "k": 5}`). |
| POST | `/api/companions` | Companion lookup (`{"plant": "tomato"}`). |
| GET | `/api/companions/plants` | Autocomplete list for companion feature. |
| POST | `/api/yield` | Yield prediction (`{"crop": "tomato", "temp": 24, "rain": 80}`). |
| POST | `/api/detect` | Crop classifier (multipart form with `image`). |
| POST | `/api/disease` | Disease classifier (multipart form with `image`, optional `notes`). |

All endpoints return standardized JSON including data payloads, confidence scores, optional KB citations, and source provenance (`knowledge_base`, `llm_knowledge`, or `knowledge_base_limited`).

## Testing & Validation

- **Unit tests (recommended additions):** focus on helper utilities such as `_check_kb_quality`, `_generate_ollama_response`, preprocessing routines, and planner fallbacks.
- **Manual validation:**
  - Run through typical chat, search, companion, yield, and detection scenarios (see `CHAT_USAGE_GUIDE.md`).
  - Confirm disease model loads successfully via `test_disease_model.py` or curl requests against `/api/disease`.
  - Validate KB ingestion by executing `kb/search.py` queries.
- **Performance expectations:** text-only queries ~1-3s, image workflows ~2-4s, depending on hardware.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: tensorflow` | TensorFlow not installed in venv | `python -m pip install tensorflow==2.16.*` |
| `/api/disease` returns 500 | Missing `.h5` model or labels | Ensure `crop_disease/trained_model_18epochs.h5` and `class_indices.pkl` exist. |
| KB search unavailable | Supabase credentials missing or FAKE_EMBEDDINGS flag set | Configure `kb/.env` and unset `$env:FAKE_EMBEDDINGS`. |
| Ollama errors (`model not found`) | `gemma3` or embedding model not pulled | Run `ollama pull gemma3:latest` (and embedding model). |
| Slow responses | Large models on CPU | Switch to quantized Ollama models, reduce top_k, or upgrade hardware. |
| Frontend layout issues | Cached assets | Hard refresh browser (Ctrl+F5) or clear cache. |

## Roadmap Ideas

- Multi-language support with automatic translation pipelines.
- Transfer-learning upgrades for image models (EfficientNet, Vision Transformers).
- Voice input/output for hands-free operation in the field.
- Automated weather + disease risk alerts based on local data.
- Mobile companion app consuming the same API endpoints.

Happy farming and experimenting! 🌱
