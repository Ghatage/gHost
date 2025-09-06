import sys
import json
import argparse
import requests
from pathlib import Path
import os

# Allow running as a script: python backend/test.py ...
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.services import _yt_id, _find_local_video, yt_video, compose_video_with_inserts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--llm", action="store_true")
    p.add_argument("--diarize", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--prompt", default="")
    p.add_argument("--compose", action="store_true", help="Compose final video by inserting AI clips at timestamps from AI_text_{id}.json")
    args = p.parse_args()
    if args.compose:
        vid = _yt_id(args.url)
        if not vid:
            print("error: could not extract video id from url", file=sys.stderr)
            sys.exit(1)
        data_dir = Path("backend/data")
        ai_fp = data_dir / f"AI_text_{vid}.json"
        if not ai_fp.exists():
            print(f"error: missing {ai_fp}", file=sys.stderr)
            sys.exit(1)
        try:
            ai = json.load(open(ai_fp, "r"))
        except Exception as e:
            print(f"error: failed to parse {ai_fp}: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(ai, list) or len(ai) < 2:
            print("error: AI_text must be a list with at least 2 entries", file=sys.stderr)
            sys.exit(1)
        try:
            t1 = float((ai[0] or {}).get("timestamp"))
            t2 = float((ai[1] or {}).get("timestamp"))
        except Exception:
            print("error: invalid timestamps in AI_text json", file=sys.stderr)
            sys.exit(1)
        clip1 = data_dir / f"AI_clip1_{vid}.mp4"
        clip2 = data_dir / f"AI_clip2_{vid}.mp4"
        if not clip1.exists() or not clip2.exists():
            print(f"error: missing clip files {clip1} or {clip2}", file=sys.stderr)
            sys.exit(1)
        main_path = _find_local_video(vid)
        if not main_path:
            print("main video not found locally; downloading...")
            main_path = yt_video(args.url)
        # Optional intro banana clip at 0s
        banana = data_dir / f"AI_banana_{vid}.mp4"
        intro = str(banana) if banana.exists() else None
        out_fp = data_dir / f"final_{vid}.mp4"
        print(f"composing -> {out_fp}")
        final_path = compose_video_with_inserts(
            main_path,
            [str(clip1), str(clip2)],
            transcript=None,
            timestamps=[t1, t2],
            out_path=str(out_fp),
            intro_path=intro,
        )
        print(f"done: {final_path}")
        return
    if args.video:
        rv = requests.post("http://127.0.0.1:8000/yt_video", json={"url": args.url}, timeout=1800)
        print("/yt_video", rv.status_code)
        print(json.dumps(rv.json(), indent=2, ensure_ascii=False))
        return
    if args.analyze:
        payload = {"url": args.url}
        if args.prompt:
            payload["prompt"] = args.prompt
        ra = requests.post("http://127.0.0.1:8000/analyze", json=payload, timeout=1800)
        print("/analyze", ra.status_code)
        data = ra.json()
        llm = (data or {}).get("llm") or {}
        if llm.get("result"):
            print("\n=== LLM Result ===\n")
            print(llm.get("result"))
            print("\n=== Full JSON ===\n")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    r = requests.post("http://127.0.0.1:8000/diarize", json={"url": args.url}, timeout=1800)
    print("/diarize", r.status_code)
    data = r.json()
    if args.diarize or not args.llm:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    segs = data.get("segments") or []
    payload2 = {"segments": segs}
    if args.prompt:
        payload2["prompt"] = args.prompt
    rr = requests.post("http://127.0.0.1:8000/llm", json=payload2, timeout=1800)
    print("/llm", rr.status_code)
    rj = rr.json()
    if rj.get("result"):
        print("\n=== LLM Result ===\n")
        print(rj.get("result"))
        print("\n=== Full JSON ===\n")
    print(json.dumps(rj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
