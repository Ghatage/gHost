import sys
import json
import argparse
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--llm", action="store_true")
    p.add_argument("--diarize", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--prompt", default="Your task is the summarize")
    args = p.parse_args()
    if args.analyze:
        ra = requests.post("http://127.0.0.1:8000/analyze", json={"url": args.url, "prompt": args.prompt}, timeout=1800)
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
    rr = requests.post("http://127.0.0.1:8000/llm", json={"segments": segs, "prompt": args.prompt}, timeout=1800)
    print("/llm", rr.status_code)
    rj = rr.json()
    if rj.get("result"):
        print("\n=== LLM Result ===\n")
        print(rj.get("result"))
        print("\n=== Full JSON ===\n")
    print(json.dumps(rj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
