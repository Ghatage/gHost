Backend (FastAPI) — Quick Start

- Setup:
  - python3 -m venv backend/.venv
  - backend/.venv/bin/pip install -r backend/requirements.txt
  - Create `backend/.env.local` with keys:
    - ASSEMBLYAI_API_KEY=your_key
    - OPENAI_API_KEY=your_key

- Run API:
  - backend/.venv/bin/uvicorn backend.routes:app --reload --port 8000

- Test (CLI):
  - Diarize only:
    - python backend/test.py <youtube_url>
    - python backend/test.py <youtube_url> --diarize
  - Diarize + LLM:
    - python backend/test.py <youtube_url> --llm --prompt "Your task is the summarize"
  - Single-step analyze (diarize + LLM):
    - python backend/test.py <youtube_url> --analyze --prompt "Your task is the summarize"

- Endpoints:
  - POST /yt_audio { url }
  - POST /diarize { url }
  - POST /llm { segments, prompt? }
  - POST /analyze { url, prompt? }

