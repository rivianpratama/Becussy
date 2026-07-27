"""Local OpenAI-compatible inference server for a Becussy checkpoint.

This is what the chat web app talks to when you want to run the model on your
own machine instead of Amazon Bedrock. The app lives in the Becussy-deploy repo
(github.com/rivianpratama/Becussy-deploy); this server is the training-side half
of that contract. It exposes the subset of the OpenAI API the app uses:

    GET  /health                 -> {"status": "ok", "model": ..., "adapter": ...}
    POST /v1/chat/completions    -> streaming (SSE) or single JSON response

Deliberately stdlib-only (http.server + threads) so it adds NO packages to the
Turing-pinned training env. Token streaming uses transformers' TextIteratorStreamer.

Prompting goes through common.infer.encode_messages, so the template and the
no-system-message convention are identical to train.py / eval / chat.py.

Usage (inside WSL, training venv active):
    python training/serve_local.py                       # latest model (v2_best)
    python training/serve_local.py --adapter ~/becussy_runs/run01/checkpoint-180
    python training/serve_local.py --port 8000 --host 0.0.0.0

Then in the web app's .env.local (Becussy-deploy repo):
    MODEL_PROVIDER=local
    LOCAL_MODEL_URL=http://localhost:8000

Note: binds 0.0.0.0 by default so Next.js running on Windows can reach it
through WSL2's localhost forwarding.
"""
from __future__ import annotations

from unsloth import FastLanguageModel  # isort: skip  (must precede torch)

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from common.infer import clean_output, encode_messages  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
INFER_CFG = CFG.get("inference") or {}
# Serving context window. Wider than the 1024 training truncation on purpose —
# that was a VRAM limit, not a model limit (native max_position_embeddings is 262144).
DEFAULT_MAX_SEQ = int(INFER_CFG.get("max_seq_length", 4096))
# 0 => generate until the model emits EOS, bounded only by remaining context.
DEFAULT_MAX_NEW = int(INFER_CFG.get("max_new_tokens", 0))

# The latest trained model: v3 — same recipe as the v2 sweep winner (r32 /
# lr1e-4 / neftune5) but trained on the v3 dataset (identity + on-topic football
# archetypes, full-corpus diversity rewrite). checkpoint-360 scored best across
# the 96-probe set: identity 8/8 with zero leaks, football leaks 5 -> 2,
# transitivity 11.5% -> 1%. It misses the 0.95 pivot_rate gate at 0.927; see
# eval/SELECTION.md for the trade-off and the known "Are you Qwen?" issue.
DEFAULT_ADAPTER = "~/becussy_runs/v3/checkpoint-360"

# Populated by main() before the server starts serving.
STATE: dict = {}
# Generation is single-GPU and not reentrant; serialize concurrent requests.
GEN_LOCK = threading.Lock()


def load_model(adapter: str | None, max_seq_length: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG["model"],
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
        revision=CFG.get("model_revision"),
    )
    if adapter:
        from peft import PeftModel

        path = os.path.expanduser(adapter)
        if not Path(path, "adapter_config.json").exists():
            sys.exit(f"no adapter_config.json in {path}")
        model = PeftModel.from_pretrained(model, path)
        print(f"loaded adapter: {path}", flush=True)
    else:
        print("base model (no adapter) — control condition", flush=True)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def normalize_messages(raw: list) -> list[dict]:
    """Keep only user/assistant turns with string content. System messages are
    dropped on purpose: the model was never trained with one."""
    out = []
    for m in raw or []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content})
    return out


def generate_stream(messages: list[dict], max_new_tokens: int, temperature: float,
                    top_p: float):
    """Yield raw text deltas from the model as they are produced.

    `max_new_tokens <= 0` means "no ceiling": generation runs until the model
    emits EOS, limited only by what is left of the context window. Generation
    always stops at EOS regardless — this is purely an upper bound.
    """
    from transformers import TextIteratorStreamer

    model, tokenizer = STATE["model"], STATE["tokenizer"]
    ids = encode_messages(tokenizer, messages)

    # Never let prompt + reply exceed the loaded context window.
    room = STATE["max_seq_length"] - int(ids.shape[1])
    if room <= 0:
        raise ValueError(
            f"prompt is {int(ids.shape[1])} tokens, which fills the "
            f"{STATE['max_seq_length']}-token context window — shorten the conversation"
        )
    budget = room if max_new_tokens <= 0 else min(max_new_tokens, room)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = dict(
        input_ids=ids,
        streamer=streamer,
        max_new_tokens=budget,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    )
    thread = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
    thread.start()
    for chunk in streamer:
        if chunk:
            yield chunk
    thread.join()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "becussy-local/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write(f"[serve_local] {fmt % args}\n")

    # --- helpers ---------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # --- routes ----------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send_json({
                "status": "ok",
                "model": STATE["model_label"],
                "adapter": STATE["adapter"] or "base",
                "max_seq_length": STATE["max_seq_length"],
                "max_new_tokens": STATE["max_new_tokens"] or "until-EOS",
            })
        elif self.path.rstrip("/") == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{"id": STATE["model_label"], "object": "model"}],
            })
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            payload = self._read_json()
        except Exception as e:
            self._send_json({"error": f"bad json: {e}"}, status=400)
            return

        messages = normalize_messages(payload.get("messages"))
        if not messages:
            self._send_json({"error": "no usable messages"}, status=400)
            return

        # Absent (or <=0) max_tokens means "let the model decide when to stop".
        raw_max = payload.get("max_tokens")
        if raw_max is None:
            raw_max = payload.get("max_gen_len")
        max_new = int(raw_max) if raw_max is not None else STATE["max_new_tokens"]
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.9))
        stream = bool(payload.get("stream", False))

        if stream:
            self._stream_response(messages, max_new, temperature, top_p)
        else:
            self._full_response(messages, max_new, temperature, top_p)

    def _stream_response(self, messages, max_new, temperature, top_p):
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()

        def frame(delta=None, finish=None):
            choice = {"index": 0, "delta": {}, "finish_reason": finish}
            if delta is not None:
                choice["delta"] = {"role": "assistant", "content": delta}
            return "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": STATE["model_label"], "choices": [choice],
            }) + "\n\n"

        try:
            with GEN_LOCK:
                for delta in generate_stream(messages, max_new, temperature, top_p):
                    self.wfile.write(frame(delta=delta).encode("utf-8"))
                    self.wfile.flush()
            self.wfile.write(frame(finish="stop").encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected / pressed Stop
        except Exception as e:
            self.log_message("generation error: %s", e)
            try:
                self.wfile.write(
                    ("data: " + json.dumps({"error": str(e)}) + "\n\n").encode("utf-8")
                )
                self.wfile.flush()
            except Exception:
                pass

    def _full_response(self, messages, max_new, temperature, top_p):
        try:
            with GEN_LOCK:
                text = "".join(generate_stream(messages, max_new, temperature, top_p))
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
            return
        text = clean_output(text)
        self._send_json({
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": STATE["model_label"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.environ.get("BECUSSY_ADAPTER", DEFAULT_ADAPTER),
                    help="checkpoint dir with the LoRA adapter, or 'base' for none")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-seq-length", type=int,
                    default=int(os.environ.get("BECUSSY_MAX_SEQ", DEFAULT_MAX_SEQ)),
                    help="serving context window (prompt + reply)")
    ap.add_argument("--max-new-tokens", type=int,
                    default=int(os.environ.get("BECUSSY_MAX_NEW", DEFAULT_MAX_NEW)),
                    help="0 = generate until EOS, bounded only by the context window")
    args = ap.parse_args()

    adapter = None if args.adapter.lower() == "base" else args.adapter
    label = "becussy-" + (Path(os.path.expanduser(adapter)).name if adapter else "base")

    print(f"loading model (adapter={adapter or 'base'}, "
          f"context={args.max_seq_length})...", flush=True)
    model, tokenizer = load_model(adapter, args.max_seq_length)
    STATE.update(model=model, tokenizer=tokenizer, adapter=adapter, model_label=label,
                 max_seq_length=args.max_seq_length,
                 max_new_tokens=args.max_new_tokens)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nBecussy local server ready on http://{args.host}:{args.port}", flush=True)
    print(f"  health : GET  /health", flush=True)
    print(f"  chat   : POST /v1/chat/completions  (stream: true supported)", flush=True)
    cap = args.max_new_tokens or "until EOS (context-bounded)"
    print(f"  context: {args.max_seq_length} tokens | reply cap: {cap}", flush=True)
    print(f"\nPoint the web app at it:  LOCAL_MODEL_URL=http://localhost:{args.port}\n",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
        srv.shutdown()


if __name__ == "__main__":
    main()
