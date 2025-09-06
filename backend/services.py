from yt_dlp import YoutubeDL
from pathlib import Path
import subprocess
import os
from dotenv import load_dotenv
import json
import tempfile


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


def _find_local_video(vid: str | None) -> str | None:
    if not vid:
        return None
    d = Path("backend/data")
    # Include common video and audio extensions; we only need audio for diarization
    for ext in ("mp4", "mkv", "webm", "m4a", "mp3", "wav"):
        p = d / f"{vid}.{ext}"
        if p.exists():
            return str(p)
    return None


def yt_video(url: str) -> str:
    print("yt_video:start", url, flush=True)
    d = Path("backend/data")
    d.mkdir(parents=True, exist_ok=True)
    vid = _yt_id(url)
    cached = _find_local_video(vid)
    if cached:
        print("yt_video:cache_hit", cached, flush=True)
        return cached
    # Pick a smaller download by default to save time/space
    quality = os.getenv("YTDL_QUALITY", "360p").lower().strip()
    fmt = None
    merge_fmt = None
    if quality in ("audio", "audio-only", "audio_only"):
        # Audio only (preferred m4a)
        fmt = "bestaudio[ext=m4a]/bestaudio/best"
        merge_fmt = None
    else:
        height_map = {"144p": 144, "240p": 240, "360p": 360, "480p": 480, "720p": 720}
        h = height_map.get(quality, 360)
        # Try to pick a video stream up to the target height with audio; fallback to best within limit; then best overall
        fmt = f"bv*[height<={h}]+ba/best[height<={h}]/best"
        merge_fmt = "mp4"
    ydl_opts = {
        "quiet": True,
        "noplaylist": True,
        "format": fmt,
        "outtmpl": str(d / "%(id)s.%(ext)s"),
    }
    if merge_fmt:
        ydl_opts["merge_output_format"] = merge_fmt
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    # Try to read the exact download file path
    try:
        reqs = info.get("requested_downloads") or []
        if reqs:
            fp = reqs[0].get("filepath")
            if fp and Path(fp).exists():
                print("yt_video:done", fp, flush=True)
                return fp
    except Exception:
        pass
    vid = vid or info.get("id")
    # Fallback: look for common output extensions
    prefs = ["mp4", "mkv", "webm", "m4a", "mp3", "wav"]
    for ext in prefs:
        p = d / f"{vid}.{ext}"
        if p.exists():
            print("yt_video:done", str(p), flush=True)
            return str(p)
    fp = info.get("_filename")
    if fp and Path(fp).exists():
        print("yt_video:done", fp, flush=True)
        return fp
    raise RuntimeError("video_file_not_found")


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
            cached = json.load(f)
        vpath = _find_local_video(vid) or yt_video(url)
        cached["video_path"] = vpath
        return cached
    print("cache:miss", vid or "unknown", flush=True)
    aai.settings.api_key = key
    vpath = _find_local_video(vid) or yt_video(url)
    try:
        path = _to_wav_16k_mono(vpath)
    except Exception:
        path = vpath
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
    out = {"text": text, "segments": segments, "video_path": vpath}
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
    prompt = custom_prompt or (
        """
You are provided with a conversation between two people in a podcast. You are an AI Assistant who will be added to the video to explain complex subjects that the listener might struggle with. You cannot ask questions or interact with the speakers—you can only add short, helpful explanations after someone finishes talking.

Your output must always include exactly 2 entries, in JSON format, with:
- "timestamp": the precise second taken from a TURN END TIME (the end value of a turn shown in [start-end]). Use only one of the listed turn end times, exactly.
- "timecode": the same timestamp formatted as HH:MM:SS (zero-padded, e.g., 00:03:07).
- "explanation": a simple, digestible one-sentence explanation that makes the subject easier to understand, containing 15–20 words. Do not go under 15 or over 20 words.

Tone and style:
- Speak in the voice of the show’s host, addressing the audience directly.
- Do not refer to speakers by role or pronoun. Avoid words like "he", "she", "they", "guest", or "host". Focus on the concept itself (e.g., "In short, X means…", "Simply put, …").
- Keep it neutral, friendly, and concise—no meta commentary.

Rules for choosing timestamps:
- Never choose a time inside a turn. Treat consecutive segments by the same speaker as one continuous turn.
- Prefer boundaries where the next speaker begins with a question. Place the explanation right after the previous speaker finishes and just before the next question.
- If no clear question boundary exists, choose a natural speaker-change boundary after the previous speaker finishes.

Return only the JSON—nothing else.
""".strip()
    )

    # Merge consecutive segments by the same speaker into single "turns"
    turns = []
    cur = None
    for s in segments or []:
        sp = s.get("speaker") or "SPEAKER"
        st = s.get("start")
        en = s.get("end")
        tx = (s.get("text") or "").strip()
        if cur and cur["speaker"] == sp:
            # extend current turn
            try:
                cur["end"] = float(en)
            except Exception:
                cur["end"] = en
            if tx:
                cur["text"] += (" " if cur["text"] else "") + tx
        else:
            # push previous
            if cur:
                turns.append(cur)
            try:
                st_f = float(st)
                en_f = float(en)
            except Exception:
                st_f = st
                en_f = en
            cur = {"speaker": sp, "start": st_f, "end": en_f, "text": tx}
    if cur:
        turns.append(cur)

    # Build turn-based transcript lines
    lines = []
    for t in turns:
        st = t.get("start")
        en = t.get("end")
        sp = t.get("speaker")
        tx = t.get("text") or ""
        try:
            lines.append(f"[{float(st):.3f}-{float(en):.3f}] {sp}: {tx}")
        except Exception:
            lines.append(f"[{st}-{en}] {sp}: {tx}")

    content = prompt + "\n\nTranscript (turns):\n" + "\n".join(lines)
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


def tts_elevenlabs_from_explanations(explanations: list, out_path: str) -> str:
    """
    Synthesize speech from a list of explanation dicts and save to out_path.
    Env vars:
      - ELEVENLABS_API_KEY (required)
      - ELEVENLABS_VOICE_ID (optional; default sample voice)
      - ELEVENLABS_MODEL_ID (optional; default eleven_multilingual_v2)
      - ELEVENLABS_OUTPUT_FORMAT (optional; default wav_44100_16bit)
    Returns out_path on success.
    """
    load_dotenv(dotenv_path=Path(__file__).with_name('.env.local'))
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("missing ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    # Use a widely compatible default; many players support MP3 reliably
    output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    try:
        from elevenlabs.client import ElevenLabs
    except Exception as e:
        raise RuntimeError(f"elevenlabs_import_failed: {e}")

    # Join explanations into a single narration
    lines = []
    for item in explanations or []:
        txt = (item or {}).get("explanation")
        if txt:
            lines.append(str(txt).strip())
    text = "\n\n".join(lines).strip()
    if not text:
        raise RuntimeError("no_explanations_to_synthesize")

    client = ElevenLabs(api_key=key)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
    )

    # Decide file extension from format (override given out_path ext if mismatched)
    fmt = (output_format or "").lower()
    if fmt.startswith("mp3"):
        ext = ".mp3"
    elif fmt.startswith("wav"):
        ext = ".wav"
    else:
        # Fallback to mp3 extension for unknown formats
        ext = ".mp3"

    out_p = Path(out_path)
    if out_p.suffix.lower() != ext:
        out_p = out_p.with_suffix(ext)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        try:
            for chunk in audio:
                if isinstance(chunk, (bytes, bytearray)):
                    f.write(chunk)
                else:
                    data = getattr(chunk, "content", None) or getattr(chunk, "data", None)
                    if data:
                        f.write(data)
        except TypeError:
            # Not iterable; assume bytes-like
            f.write(audio)
    return str(out_p)


def _ffprobe_duration(path: str) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return float(r.stdout.decode().strip())
    except Exception as e:
        raise RuntimeError(f"ffprobe_failed: {e}")


def _ffprobe_video_props(path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            path,
        ]
        r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = r.stdout.decode().strip()
        w, h = out.split("x")
        return int(w), int(h)
    except Exception as e:
        raise RuntimeError(f"ffprobe_props_failed: {e}")


def _has_audio(path: str) -> bool:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            path,
        ]
        r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return bool(r.stdout.decode().strip())
    except Exception:
        return False


def _ffmpeg_add_silent_audio(video_in: str, duration: float, out_path: str) -> str:
    """Mux a silent stereo track to a video-only input to ensure it has audio."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_in,
        "-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "44100",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def _ffmpeg_trim(input_path: str, start: float, end: float, out_path: str) -> str:
    """Trim input between start and end (seconds) with accurate seek, re-encode for reliability."""
    if end <= start:
        raise ValueError("end_must_be_gt_start")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "44100",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Ensure audio exists; if missing, add silent track
        if not _has_audio(out_path):
            dur = max(0.0, end - start)
            tmp = str(Path(out_path).with_suffix(".tmp.mp4"))
            _ffmpeg_add_silent_audio(out_path, dur, tmp)
            os.replace(tmp, out_path)
        return out_path
    except Exception as e:
        raise RuntimeError(f"ffmpeg_trim_failed: {e}")


def _ffmpeg_normalize_clip(input_path: str, width: int, height: int, out_path: str) -> str:
    """Transcode a clip to match target width/height, stereo 44.1kHz, H.264/AAC, letterboxed if needed."""
    vf = f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "44100",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Ensure an audio stream exists; add silent if needed
        if not _has_audio(out_path):
            dur = _ffprobe_duration(input_path)
            tmp = str(Path(out_path).with_suffix(".tmp.mp4"))
            _ffmpeg_add_silent_audio(out_path, dur, tmp)
            os.replace(tmp, out_path)
        return out_path
    except Exception as e:
        raise RuntimeError(f"ffmpeg_normalize_failed: {e}")


def compose_video_with_inserts(main_path: str, clip_paths: list, transcript: list | None, timestamps: list | None, out_path: str, intro_path: str | None = None) -> str:
    """
    Create a final MP4 by inserting exactly two clip videos into a main video at transcript timestamps.

    - main_path: path to the base MP4
    - clip_paths: list of 2 MP4 paths to insert (order corresponds to timestamps)
    - transcript: list of dicts with at least a numeric 'timestamp' field; first two are used if timestamps not provided
    - timestamps: optional list of 2 floats (seconds) where to insert clips into the main video
    - out_path: final MP4 path

    Strategy: split main into [0:t1], [t1:t2], [t2:end], then concat:
      (optional intro), part1, clip1, part2, clip2, part3.
    Re-encodes to H.264/AAC for consistency.
    """
    if not isinstance(clip_paths, list) or len(clip_paths) != 2:
        raise ValueError("clip_paths_must_have_two_items")
    for p in [main_path] + clip_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"path_not_found: {p}")

    # Determine insertion times (seconds)
    ts = []
    if timestamps and len(timestamps) == 2:
        ts = [float(timestamps[0]), float(timestamps[1])]
    else:
        if not transcript or len(transcript) < 2:
            raise ValueError("need_two_timestamps_or_transcript_with_two_entries")
        for i in range(2):
            item = transcript[i] or {}
            t = item.get("timestamp")
            if t is None:
                raise ValueError("transcript_missing_timestamp")
            try:
                ts.append(float(t))
            except Exception:
                raise ValueError("invalid_timestamp_value")

    # sort by time but keep mapping to clip_paths
    pairs = sorted(zip(ts, clip_paths), key=lambda x: x[0])
    t1, clip1 = pairs[0]
    t2, clip2 = pairs[1]
    if t2 <= t1:
        raise ValueError("timestamps_must_be_increasing")

    dur = _ffprobe_duration(main_path)
    if t1 <= 0 or t1 >= dur:
        raise ValueError("t1_out_of_bounds")
    if t2 <= 0 or t2 >= dur:
        raise ValueError("t2_out_of_bounds")

    with tempfile.TemporaryDirectory() as td:
        part1 = os.path.join(td, "part1.mp4")
        part2 = os.path.join(td, "part2.mp4")
        part3 = os.path.join(td, "part3.mp4")
        _ffmpeg_trim(main_path, 0.0, t1, part1)
        _ffmpeg_trim(main_path, t1, t2, part2)
        _ffmpeg_trim(main_path, t2, dur, part3)

        # Normalize insert clips to match main resolution and audio format
        w, h = _ffprobe_video_props(main_path)
        # Optional intro clip at time 0
        intro_norm = None
        if intro_path and os.path.exists(intro_path):
            intro_norm = os.path.join(td, "intro_norm.mp4")
            _ffmpeg_normalize_clip(intro_path, w, h, intro_norm)
        norm1 = os.path.join(td, "clip1_norm.mp4")
        norm2 = os.path.join(td, "clip2_norm.mp4")
        _ffmpeg_normalize_clip(clip1, w, h, norm1)
        _ffmpeg_normalize_clip(clip2, w, h, norm2)

        # Concat inputs with concat filter (all now aligned)
        inputs = [part1, norm1, part2, norm2, part3]
        if intro_norm:
            inputs = [intro_norm] + inputs
        cmd = ["ffmpeg", "-y"]
        for p in inputs:
            cmd += ["-i", p]
        # Build filter: [0:v][0:a][1:v][1:a]...[4:v][4:a]concat=n=5:v=1:a=1[v][a]
        n = len(inputs)
        in_labels = "".join([f"[{i}:v][{i}:a]" for i in range(n)])
        filter_str = f"{in_labels}concat=n={n}:v=1:a=1[v][a]"
        cmd += [
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg_concat_failed: {e.stderr.decode(errors='ignore')[:2000]}")
    return out_path
