from yt_dlp import YoutubeDL
from pathlib import Path
import subprocess
import os
from dotenv import load_dotenv
import json


def yt_audio(url: str) -> str:
    print("yt_audio:start", url, flush=True)
    d = Path("backend/data")
    d.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "quiet": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": str(d / "%(id)s.%(ext)s"),
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    fp = info["requested_downloads"][0]["filepath"]
    print("yt_audio:done", fp, flush=True)
    return fp


def _yt_id(url: str) -> str | None:
    try:
        with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("id")
    except Exception:
        return None


def _to_wav_16k_mono(src_path: str) -> str:
    print("ffmpeg:start", src_path, flush=True)
    p = Path(src_path)
    out = p.with_suffix(".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(p),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpeg:done", str(out), flush=True)
        return str(out)
    except Exception as e:
        raise RuntimeError(f"ffmpeg_convert_failed: {e}")


async def diarize_youtube(url: str, num_speakers: int = 2) -> dict:
    print("diarize:start", url, flush=True)
    import assemblyai as aai
    load_dotenv(dotenv_path=Path(__file__).with_name('.env.local'))
    key = os.getenv("ASSEMBLYAI_API_KEY")
    if not key:
        raise RuntimeError("missing ASSEMBLYAI_API_KEY")
    vid = _yt_id(url)
    cache_dir = Path("backend/data/diarize_data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_fp = cache_dir / f"{vid}.json" if vid else None
    if cache_fp and cache_fp.exists():
        print("cache:hit", str(cache_fp), flush=True)
        with open(cache_fp, "r") as f:
            return json.load(f)
    print("cache:miss", vid or "unknown", flush=True)
    aai.settings.api_key = key
    path = yt_audio(url)
    try:
        path = _to_wav_16k_mono(path)
    except Exception:
        pass
    print("aai:transcribe:start", path, flush=True)
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        format_text=True,
        punctuate=True,
        speech_model=aai.SpeechModel.universal,
        language_code="en_us",
    )
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(path)
    if getattr(transcript, "status", None) == getattr(aai, "TranscriptStatus").error:
        raise RuntimeError(str(transcript.error))
    print("aai:transcribe:done", flush=True)
    text = transcript.text or ""
    segments = []
    uts = getattr(transcript, "utterances", None)
    if uts:
        for u in uts:
            segments.append({
                "speaker": getattr(u, "speaker", None),
                "start": (getattr(u, "start", 0) or 0)/1000.0,
                "end": (getattr(u, "end", 0) or 0)/1000.0,
                "text": getattr(u, "text", "") or "",
            })
    out = {"text": text, "segments": segments}
    if cache_fp:
        try:
            with open(cache_fp, "w") as f:
                json.dump(out, f)
            print("cache:save", str(cache_fp), flush=True)
        except Exception:
            pass
    return out


def llm_process(segments: list, custom_prompt: str | None = None) -> dict:
    load_dotenv(dotenv_path=Path(__file__).with_name('.env.local'))
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("missing OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    prompt = custom_prompt or "Your task is the summarize"
    lines = []
    for s in segments or []:
        st = s.get("start")
        en = s.get("end")
        sp = s.get("speaker") or "SPEAKER"
        tx = s.get("text", "")
        try:
            lines.append(f"[{float(st):.3f}-{float(en):.3f}] {sp}: {tx}")
        except Exception:
            lines.append(f"[{st}-{en}] {sp}: {tx}")
    content = prompt + "\n\nTranscript:\n" + "\n".join(lines)
    client = OpenAI(api_key=key)
    r = client.responses.create(
        model=model,
        input=content,
        text={"format": {"type": "text"}, "verbosity": "low"},
        reasoning={"effort": "low", "summary": "auto"},
        tools=[],
        store=True,
        include=[
            "reasoning.encrypted_content",
            "web_search_call.action.sources",
        ],
    )
    out = getattr(r, "output_text", "") or str(r)
    return {"result": out, "model": model}
