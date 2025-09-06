from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .services import yt_audio, diarize_youtube, llm_process


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UrlIn(BaseModel):
    url: str


@app.post("/yt_audio")
def download_audio(body: UrlIn):
    try:
        path = yt_audio(body.url)
        return {"path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diarize")
async def diarize(body: UrlIn):
    try:
        r = await diarize_youtube(body.url, 2)
        return r
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LLMIn(BaseModel):
    segments: list
    prompt: str | None = None


@app.post("/llm")
def llm(body: LLMIn):
    try:
        r = llm_process(body.segments, body.prompt)
        return r
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AnalyzeIn(BaseModel):
    url: str
    prompt: str | None = None


@app.post("/analyze")
async def analyze(body: AnalyzeIn):
    try:
        d = await diarize_youtube(body.url, 2)
        r = llm_process(d.get("segments") or [], body.prompt)
        return {"diarization": d, "llm": r}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
