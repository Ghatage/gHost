from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .services import yt_audio, yt_video, diarize_youtube, llm_process, _to_wav_16k_mono, _yt_id
from .services import compose_video_with_inserts
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
import json, threading, queue, os
import time

PHASES = {
    "init": "Initiating gHost Mode",
    "process_video": "Processing video",
    "diarize": "Diarizing",
    "find_frames": "Finding the right frames to gHost",
    "summary": "Summary",
    "compose": "Composing final video",
}


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


@app.post("/yt_video")
def download_video(body: UrlIn):
    try:
        path = yt_video(body.url)
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
        # Persist AI explanations if possible
        try:
            vid = _yt_id(body.url)
            if vid:
                out_dir = os.path.join("backend", "data")
                os.makedirs(out_dir, exist_ok=True)
                fp = os.path.join(out_dir, f"AI_text_{vid}.json")
                res_text = (r or {}).get("result") or ""
                parsed = json.loads(res_text)
                with open(fp, "w") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)
                print(f"[analyze] ai_text:save {fp}", flush=True)
        except Exception:
            print("[analyze] ai_text:error", flush=True)
        return {"diarization": d, "llm": r}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_stream")
async def analyze_stream(body: AnalyzeIn):
    q = queue.Queue()

    def emit(obj):
        try:
            q.put(json.dumps(obj) + "\n")
        except Exception:
            pass

    def stage(phase: str, status: str):
        emit({"type": "stage", "phase": phase, "status": status, "label": PHASES.get(phase, phase)})

    def work():
        try:
            stage("init", "started")
            vid = _yt_id(body.url)
            cache_dir = os.path.join("backend", "data", "diarize_data")
            os.makedirs(cache_dir, exist_ok=True)
            cache_fp = os.path.join(cache_dir, f"{vid}.json") if vid else None
            if cache_fp and os.path.exists(cache_fp):
                emit({"type": "log", "stage": "cache_hit", "path": cache_fp})
                stage("init", "completed")
                # Download/ensure video saved
                stage("process_video", "started")
                vpath = None
                vid = _yt_id(body.url)
                from .services import _find_local_video
                vpath = _find_local_video(vid)
                if vpath:
                    emit({"type": "log", "stage": "video_cache_hit", "path": vpath})
                else:
                    vpath = yt_video(body.url)
                stage("process_video", "completed")
                # Mark diarize as completed due to cache hit
                stage("diarize", "completed")
                with open(cache_fp, "r") as f:
                    diar = json.load(f)
                diar["video_path"] = vpath
            else:
                stage("init", "completed")
                stage("process_video", "started")
                emit({"type": "log", "stage": "download:start"})
                vid = _yt_id(body.url)
                from .services import _find_local_video
                vpath = _find_local_video(vid)
                if vpath:
                    emit({"type": "log", "stage": "video_cache_hit", "path": vpath})
                else:
                    vpath = yt_video(body.url)
                    emit({"type": "log", "stage": "download:done", "path": vpath})
                try:
                    emit({"type": "log", "stage": "convert:start"})
                    apath = _to_wav_16k_mono(vpath)
                    emit({"type": "log", "stage": "convert:done", "path": apath})
                except Exception as e:
                    emit({"type": "log", "stage": "convert:skip", "error": str(e)})
                    apath = vpath
                import assemblyai as aai
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=os.path.join("backend", ".env.local"))
                key = os.getenv("ASSEMBLYAI_API_KEY")
                if not key:
                    raise RuntimeError("missing ASSEMBLYAI_API_KEY")
                aai.settings.api_key = key
                stage("process_video", "completed")
                stage("diarize", "started")
                emit({"type": "log", "stage": "diarize:start"})
                config = aai.TranscriptionConfig(
                    speaker_labels=True,
                    format_text=True,
                    punctuate=True,
                    speech_model=aai.SpeechModel.universal,
                    language_code="en_us",
                )
                transcriber = aai.Transcriber(config=config)
                tr = transcriber.transcribe(apath)
                if getattr(tr, "status", None) == getattr(aai, "TranscriptStatus").error:
                    raise RuntimeError(str(tr.error))
                emit({"type": "log", "stage": "diarize:done"})
                stage("diarize", "completed")
                text = tr.text or ""
                segs = []
                uts = getattr(tr, "utterances", None)
                if uts:
                    for u in uts:
                        segs.append({
                            "speaker": getattr(u, "speaker", None),
                            "start": (getattr(u, "start", 0) or 0)/1000.0,
                            "end": (getattr(u, "end", 0) or 0)/1000.0,
                            "text": getattr(u, "text", "") or "",
                        })
                diar = {"text": text, "segments": segs, "video_path": vpath}
                if cache_fp:
                    try:
                        with open(cache_fp, "w") as f:
                            json.dump(diar, f)
                        emit({"type": "log", "stage": "cache:save", "path": cache_fp})
                    except Exception:
                        pass
            stage("find_frames", "started")
            time.sleep(0.1)
            stage("find_frames", "completed")
            stage("summary", "started")
            emit({"type": "log", "stage": "llm:start"})
            llm = llm_process(diar.get("segments") or [], body.prompt)
            emit({"type": "log", "stage": "llm:done"})
            stage("summary", "completed")
            # Save AI explanations to file for later use
            try:
                if vid:
                    out_dir = os.path.join("backend", "data")
                    os.makedirs(out_dir, exist_ok=True)
                    fp = os.path.join(out_dir, f"AI_text_{vid}.json")
                    res_text = (llm or {}).get("result") or ""
                    parsed = json.loads(res_text)
                    with open(fp, "w") as f:
                        json.dump(parsed, f, ensure_ascii=False, indent=2)
                    emit({"type": "log", "stage": "ai_text:save", "path": fp})
                    # Compose final video with available AI_clip{i}_{vid}.mp4 files at timestamps from AI_text JSON
                    final_fp = os.path.join(out_dir, f"final_{vid}.mp4")
                    try:
                        stage("compose", "started")
                        # If final already exists, reuse it and skip composing
                        if os.path.exists(final_fp):
                            emit({"type": "log", "stage": "compose:reuse", "path": final_fp})
                        else:
                            # Collect up to 4 clip files, ordered by index
                            clip_candidates = [os.path.join(out_dir, f"AI_clip{i}_{vid}.mp4") for i in range(1, 5)]
                            clips_found = [p for p in clip_candidates if os.path.exists(p)]
                            # Optional intro banana clip at t=0
                            intro_path = os.path.join(out_dir, f"AI_banana_{vid}.mp4")
                            intro_exists = os.path.exists(intro_path)
                            emit({"type": "log", "stage": "compose:found", "clips": clips_found, "intro": intro_path if intro_exists else None})
                            # Build timestamps list from parsed JSON
                            ts = []
                            if isinstance(parsed, list):
                                for item in parsed:
                                    try:
                                        ts.append(float((item or {}).get("timestamp")))
                                    except Exception:
                                        pass
                            emit({"type": "log", "stage": "compose:inputs", "timestamps_found": len(ts), "clips_found": len(clips_found)})
                            # Use the first two available clips and timestamps
                            if len(clips_found) >= 2 and len(ts) >= 2:
                                clips_use = clips_found[:2]
                                ts_use = ts[:2]
                                emit({"type": "log", "stage": "compose:start", "clips": clips_use, "timestamps": ts_use, "output": final_fp, "intro": intro_path if intro_exists else None})
                                main_path = (diar or {}).get("video_path")
                                if not main_path:
                                    # Try to resolve main video again if missing
                                    from .services import _find_local_video
                                    main_path = _find_local_video(vid) or vpath
                                compose_video_with_inserts(main_path, clips_use, transcript=None, timestamps=ts_use, out_path=final_fp, intro_path=intro_path if intro_exists else None)
                                emit({"type": "log", "stage": "compose:done", "path": final_fp})
                            else:
                                emit({"type": "log", "stage": "compose:skip", "reason": "insufficient_clips_or_timestamps"})
                    except Exception as e:
                        emit({"type": "log", "stage": "compose:error", "error": str(e)})
                    finally:
                        stage("compose", "completed")
                        # Always emit a final message so the popup can finish the flow
                        final_url = f"http://127.0.0.1:8000/final_video/{vid}"
                        final_exists = os.path.exists(final_fp)
                        emit({"type": "final", "video_id": vid, "final_path": final_fp if final_exists else None, "final_url": final_url if final_exists else None})
            except Exception as e:
                emit({"type": "log", "stage": "ai_text:error", "error": str(e)})
            emit({"type": "result", "diarization": diar, "llm": llm})
        except Exception as e:
            emit({"type": "error", "error": str(e)})
        finally:
            q.put(None)

    t = threading.Thread(target=work, daemon=True)
    t.start()

    async def gen():
        while True:
            item = await run_in_threadpool(q.get)
            if item is None:
                break
            try:
                # Mirror stream events to server logs for visibility
                print(f"[analyze_stream] {item.strip()}", flush=True)
            except Exception:
                pass
            yield item

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/final_video/{vid}")
def get_final_video(vid: str, request: Request):
    """Serve the composed final video with basic Range support for playback."""
    import os
    path = os.path.join("backend", "data", f"final_{vid}.mp4")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="final_video_not_found")
    file_size = os.path.getsize(path)
    range_header = request.headers.get('range')
    if range_header:
        try:
            bytes_unit, byte_range = range_header.split('=')
            start_s, end_s = byte_range.split('-')
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            start = max(0, start)
            end = min(file_size - 1, end)
            if start > end:
                start, end = 0, file_size - 1
            length = end - start + 1
            with open(path, 'rb') as f:
                f.seek(start)
                data = f.read(length)
            headers = {
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(length),
                'Content-Type': 'video/mp4',
            }
            return Response(content=data, status_code=206, headers=headers)
        except Exception:
            pass
    # Fallback full response
    with open(path, 'rb') as f:
        data = f.read()
    headers = {
        'Accept-Ranges': 'bytes',
        'Content-Length': str(len(data)),
        'Content-Type': 'video/mp4',
    }
    return Response(content=data, status_code=200, headers=headers)
