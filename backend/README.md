Backend (FastAPI) — Quick Start

- Setup:
  - python3 -m venv backend/.venv
  - backend/.venv/bin/pip install -r backend/requirements.txt
  - Create `backend/.env.local` with keys:
    - ASSEMBLYAI_API_KEY=your_key
    - OPENAI_API_KEY=your_key
    - YTDL_QUALITY=360p  # optional: audio | 144p | 240p | 360p | 480p | 720p
    - ELEVENLABS_API_KEY=your_key  # optional: enable AI audio narration
    - ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb  # optional
    - ELEVENLABS_MODEL_ID=eleven_multilingual_v2  # optional
    - ELEVENLABS_OUTPUT_FORMAT=mp3_44100_128  # optional; defaults to mp3 for broad compatibility

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

Notes:
- Download quality: Set `YTDL_QUALITY` in `.env.local` to control the size/speed of the YouTube download used for processing. `audio` downloads audio-only; otherwise the value caps video height (e.g., `360p`).
- AI narration: If ELEVENLABS_API_KEY is set, the backend will synthesize an audio narration of the explanations and save it as `backend/data/AI_Audio_{video_id}.mp3` (or `.wav` depending on ELEVENLABS_OUTPUT_FORMAT) after LLM generation in both streaming and non-streaming flows.
