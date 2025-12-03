# AI Master Gardener – Frontend

This folder contains a local-first Flask app that provides the UI for your assistant: Chat, Search, Analyze, Companions, Yield, Feedback, and Settings. It supports a Mock mode so you can demo the full UX without backend dependencies.

## Quick start (Windows PowerShell)

Ensure your venv is already created and activated (Python 3.11). Then:

```powershell
# Install UI dependencies (once)
& "c:\Users\Giri prasad\Desktop\capstone-Project\.venv311\Scripts\python.exe" -m pip install -r "c:\Users\Giri prasad\Desktop\capstone-Project\app\requirements.txt"

# Optional: smoke test with fake embeddings (for KB Search)
$env:FAKE_EMBEDDINGS = "1"

# Run the app
& "c:\Users\Giri prasad\Desktop\capstone-Project\.venv311\Scripts\python.exe" "c:\Users\Giri prasad\Desktop\capstone-Project\app\app.py"
```

Open http://127.0.0.1:8000

## Features

- Chat: message history, streaming mock replies, Clear button, persisted in localStorage.
- Search: calls live `/api/search` when Mock mode is off; returns KB chunks from Supabase.
- Analyze: image upload + notes; shows mock diagnosis now (live wiring later).
- Companions: simple mock suggestions (good/avoid) by plant.
- Yield: simple mock estimator by crop + temp + rainfall.
- Feedback: thumbs + notes saved locally.
- Settings: toggle "Use mock data" (on by default).

## Live integration (later)

For Search to call Supabase:
- Configure `kb/.env` (SUPABASE_URL and key). The Flask app reads it.
- Disable Mock mode in Settings to use the live `/api/search` endpoint.

For Analyze, Companions, Yield, and Chat:
- We'll add API routes to call your trained models and LLM via local Ollama.
- UI is ready; switching Mock off will enable real responses once endpoints exist.

## Troubleshooting

- If the page loads but Search errors, check environment variables in `kb/.env` and ensure your Supabase tables/functions exist.
- If you want a purely mock demo, keep Mock mode ON in Settings and set `$env:FAKE_EMBEDDINGS = "1"`.
- Static assets are under `app/static/`; HTML template is `app/templates/index.html`.
