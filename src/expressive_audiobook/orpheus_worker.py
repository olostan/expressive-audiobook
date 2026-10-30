"""Runs inside the Orpheus virtual environment (mlx-audio installed). Not imported by the main program.

Protocol, one JSON object per line on stdin/stdout:
  start:  {"ready": true, "sample_rate": 24000}   or   {"error": "..."}
  in:     {"text": ..., "voice": ..., "temperature": ..., "top_p": ..., "repetition_penalty": ..., "max_tokens": ...}
  out:    {"ok": true, "pcm": "<base64 16-bit mono>", "sample_rate": 24000}   or   {"ok": false, "error": "..."}
"""

import argparse
import base64
import json
import sys


def reply(value: dict) -> None:
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    # mlx-audio prints progress to stdout; keep stdout for the protocol only.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    try:
        import numpy as np
        from mlx_audio.tts.utils import load_model

        model = load_model(args.model)
    except Exception as exc:  # reported to the parent, which shows it
        sys.stdout = protocol
        reply({"error": f"could not load {args.model}: {exc}"})
        return 1
    sys.stdout = protocol
    rate = int(getattr(model, "sample_rate", 24000))
    reply({"ready": True, "sample_rate": rate})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            sys.stdout = sys.stderr
            chunks = []
            for result in model.generate(
                job["text"],
                voice=job["voice"],
                temperature=job.get("temperature", 0.6),
                top_p=job.get("top_p", 0.8),
                repetition_penalty=job.get("repetition_penalty", 1.3),
                max_tokens=job.get("max_tokens", 1200),
                split_pattern="\u0000",
            ):
                chunks.append(np.array(result.audio, dtype=np.float32).reshape(-1))
            sys.stdout = protocol
            if not chunks:
                reply({"ok": False, "error": "the model produced no audio"})
                continue
            audio = np.clip(np.concatenate(chunks), -1.0, 1.0)
            pcm = (audio * 32767.0).astype("<i2").tobytes()
            reply({"ok": True, "pcm": base64.b64encode(pcm).decode("ascii"), "sample_rate": rate})
        except Exception as exc:
            sys.stdout = protocol
            reply({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
