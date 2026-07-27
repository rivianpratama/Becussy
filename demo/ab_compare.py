"""Side-by-side A/B tester for the Becussy model generations (v1-v4).

Pick two models from the dropdowns, type one prompt, get both answers next to
each other. Blind by default: which column is which is hidden until you click
Reveal, so you judge the text rather than the label.

    python demo/ab_compare.py                    # v3 vs v4, port 8010
    python demo/ab_compare.py --a v2 --b v4
    python demo/ab_compare.py --runs-root ~/becussy_runs --port 8011

Then open http://localhost:8010

Each generation maps to one chosen checkpoint (see MODELS below). They are
all LoRA adapters over the SAME 4-bit base, so the base is
loaded once (~4.5 GB) and adapters are attached on demand and swapped with
PEFT's set_adapter(). Only ADAPTER_CACHE adapters are kept resident; the
least-recently-used is evicted after that, so all four fit inside 12 GB.

Stdlib only, matching training/serve_local.py — no new dependencies. Generation
is serialized behind a lock: one GPU, one stream at a time.
"""
from __future__ import annotations

from unsloth import FastLanguageModel  # isort: skip  (must precede torch)

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.infer import clean_output, encode_messages  # noqa: E402

CFG = yaml.safe_load((REPO / "training" / "config.yaml").read_text(encoding="utf-8"))
INFER = CFG.get("inference") or {}

BASE_ID = "__base__"
ADAPTER_CACHE = 4          # resident adapters before LRU eviction
STATE: dict = {}
GEN_LOCK = threading.Lock()

# One entry per model generation — the best checkpoint of each run, not the
# whole checkpoint tree. Provenance of each pick:
#   v1  run01/checkpoint-240  — the v1 candidate in eval/compare.py
#   v2  v2_best/checkpoint-300 — sweep winner, was the pinned production model
#   v3  v3/checkpoint-360     — best across the 96-probe set (eval/SELECTION.md)
#   v4  v4/checkpoint-360     — chosen by hand; v4's later checkpoints have no
#                               eval yet (generations exist only for 60-180),
#                               so this is a judgement call, not a measured
#                               winner. It does mirror v3, whose winner was
#                               also step 360. Update once v4 is evaluated.
MODELS = [
    {"id": "run01/checkpoint-240", "label": "v1"},
    {"id": "v2_best/checkpoint-300", "label": "v2"},
    {"id": "v3/checkpoint-360", "label": "v3"},
    {"id": "v4/checkpoint-360", "label": "v4"},
]


def ensure_adapter(model, mid: str) -> None:
    """Attach `mid` if it isn't resident; evict the LRU adapter past the cap."""
    lru: list[str] = STATE["lru"]
    if mid in lru:
        lru.remove(mid)
        lru.append(mid)
        return
    path = str(STATE["runs_root"] / mid)
    name = mid.replace("/", "__")
    t0 = time.time()
    model.load_adapter(path, adapter_name=name)
    lru.append(mid)
    print(f"loaded {mid} ({time.time() - t0:.1f}s, resident={len(lru)})", flush=True)
    while len(lru) > ADAPTER_CACHE:
        drop = lru.pop(0)
        try:
            model.delete_adapter(drop.replace("/", "__"))
            print(f"evicted {drop}", flush=True)
        except Exception as e:  # noqa: BLE001 — eviction is best-effort
            print(f"could not evict {drop}: {e}", flush=True)


def generate(mid: str, prompt: str, temperature: float, max_new_tokens: int) -> dict:
    model, tokenizer = STATE["model"], STATE["tokenizer"]
    with GEN_LOCK:
        if mid != BASE_ID:
            ensure_adapter(model, mid)
            model.set_adapter(mid.replace("/", "__"))
        ids = encode_messages(tokenizer, [{"role": "user", "content": prompt}])
        room = STATE["max_seq_length"] - int(ids.shape[1])
        if room <= 0:
            return {"text": "[prompt fills the whole context window]",
                    "secs": 0.0, "tokens": 0}
        budget = room if max_new_tokens <= 0 else min(max_new_tokens, room)
        kwargs = dict(
            input_ids=ids, max_new_tokens=budget,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=0.9)
        t0 = time.time()
        if mid == BASE_ID:
            with model.disable_adapter():
                out = model.generate(**kwargs)
        else:
            out = model.generate(**kwargs)
        new = out[0][ids.shape[1]:]
        return {
            "text": clean_output(tokenizer.decode(new, skip_special_tokens=True)),
            "secs": round(time.time() - t0, 1),
            "tokens": int(new.shape[0]),
        }


PAGE = """<!doctype html><meta charset="utf-8">
<title>Becussy A/B</title>
<style>
 :root{color-scheme:light dark}
 body{font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;
      max-width:1140px;margin:0 auto;padding:24px}
 h1{font-size:19px;margin:0 0 4px}
 .sub{opacity:.65;font-size:13px;margin-bottom:16px}
 textarea{width:100%;min-height:80px;padding:11px;font:inherit;border-radius:9px;
      border:1px solid #8884;background:transparent;color:inherit;resize:vertical}
 select{font:inherit;padding:6px 8px;border-radius:7px;border:1px solid #8886;
      background:transparent;color:inherit;max-width:270px}
 .row{display:flex;gap:14px;align-items:center;margin:12px 0;flex-wrap:wrap}
 button{font:inherit;padding:8px 18px;border-radius:8px;border:1px solid #8886;
      background:#8881;color:inherit;cursor:pointer}
 button:hover:not(:disabled){background:#8883}
 button:disabled{opacity:.45;cursor:default}
 label{font-size:13px;opacity:.85;display:flex;gap:6px;align-items:center}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px}
 @media(max-width:780px){.cols{grid-template-columns:1fr}}
 .card{border:1px solid #8884;border-radius:11px;padding:14px;min-height:120px}
 .tag{font-weight:600;font-size:13px;margin-bottom:9px;display:flex;
      justify-content:space-between;gap:8px}
 .meta{opacity:.55;font-weight:400;font-size:12px}
 .txt{white-space:pre-wrap}
 .hint{opacity:.55;font-size:12.5px;margin-top:16px}
 .err{color:#c33}
</style>
<h1>Becussy A/B</h1>
<div class="sub">Pick two model versions, one prompt, both answers. Blind until you reveal.</div>

<div class="row">
  <label>A <select id="ma"></select></label>
  <label>B <select id="mb"></select></label>
  <button id="swap" title="swap A and B">⇄</button>
</div>

<textarea id="q" placeholder="Ask something… e.g. Who are you?  /  Are you Qwen?  /  What is 17 x 24?"></textarea>

<div class="row">
  <button id="go">Compare</button>
  <button id="rev" disabled>Reveal which is which</button>
  <label>temp <input id="t" type="range" min="0" max="1.2" step="0.1" value="0.7"
      oninput="tv.textContent=this.value"><span id="tv">0.7</span></label>
  <label><input id="same" type="checkbox" checked> same seed</label>
</div>

<div class="cols">
  <div class="card"><div class="tag"><span id="n1">Left</span><span class="meta" id="m1"></span></div><div class="txt" id="o1"></div></div>
  <div class="card"><div class="tag"><span id="n2">Right</span><span class="meta" id="m2"></span></div><div class="txt" id="o2"></div></div>
</div>
<div class="hint">Sides shuffle every run. First use of a version takes a few extra seconds to attach.
Try: <i>Who are you?</i> · <i>Are you Qwen?</i> · <i>Who is the greatest footballer of all time?</i> ·
<i>Who scored Indonesia's goals against Saudi Arabia?</i></div>

<script>
let last=null;
const $=id=>document.getElementById(id);
async function init(){
  const {models,default_a,default_b}=await (await fetch('/models')).json();
  for(const sel of [$('ma'),$('mb')]){
    for(const m of models){
      const o=document.createElement('option'); o.value=m.id; o.textContent=m.label;
      o.title=m.id; sel.appendChild(o);
    }
  }
  $('ma').value=default_a; $('mb').value=default_b;
}
$('swap').onclick=()=>{ const a=$('ma').value; $('ma').value=$('mb').value; $('mb').value=a; };
$('go').onclick=async()=>{
  const q=$('q').value.trim(); if(!q)return;
  $('go').disabled=true; $('rev').disabled=true;
  $('n1').textContent='Left'; $('n2').textContent='Right';
  $('o1').className='txt'; $('o1').textContent='thinking…'; $('o2').textContent='thinking…';
  $('m1').textContent=''; $('m2').textContent='';
  try{
    const r=await fetch('/compare',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({prompt:q,a:$('ma').value,b:$('mb').value,
        temperature:parseFloat($('t').value),same_seed:$('same').checked})});
    if(!r.ok) throw new Error(await r.text());
    const d=await r.json(); last=d;
    $('o1').textContent=d.left.text;  $('m1').textContent=d.left.tokens+' tok · '+d.left.secs+'s';
    $('o2').textContent=d.right.text; $('m2').textContent=d.right.tokens+' tok · '+d.right.secs+'s';
    $('rev').disabled=false;
  }catch(e){ $('o2').textContent=''; $('o1').className='txt err'; $('o1').textContent='error: '+e.message; }
  $('go').disabled=false;
};
$('rev').onclick=()=>{ if(!last)return;
  $('n1').textContent=last.left.name; $('n2').textContent=last.right.name; };
$('q').addEventListener('keydown',e=>{ if(e.key==='Enter'&&(e.metaKey||e.ctrlKey))$('go').click(); });
init();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/models":
            self._send(200, json.dumps({
                "models": STATE["models"],
                "default_a": STATE["default_a"], "default_b": STATE["default_b"],
            }).encode(), "application/json")
        elif self.path == "/health":
            self._send(200, json.dumps({
                "status": "ok", "n_models": len(STATE["models"]),
                "resident": STATE["lru"],
            }).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/compare":
            return self._send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("content-length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            prompt = (req.get("prompt") or "").strip()
            if not prompt:
                return self._send(400, b"empty prompt", "text/plain")
            valid = {m["id"] for m in STATE["models"]}
            picks = {"A": req.get("a"), "B": req.get("b")}
            for slot, mid in picks.items():
                if mid not in valid:
                    return self._send(400, f"unknown model {mid!r}".encode(), "text/plain")
            temp = float(req.get("temperature", 0.7))
            same_seed = bool(req.get("same_seed", True))

            results = {}
            for slot, mid in picks.items():
                if same_seed:
                    torch.manual_seed(STATE["seed"])
                results[slot] = generate(mid, prompt, temp, STATE["max_new_tokens"])
                results[slot]["name"] = STATE["labels"].get(mid, mid)

            order = ["A", "B"]
            random.shuffle(order)          # blind: randomise the columns
            self._send(200, json.dumps({
                "left": results[order[0]], "right": results[order[1]],
            }).encode("utf-8"), "application/json")
        except Exception as e:  # noqa: BLE001 — surface errors to the page
            self._send(500, f"{type(e).__name__}: {e}".encode(), "text/plain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="~/becussy_runs")
    ap.add_argument("--a", default=None, help="default A: a label (v1..v4) or full path")
    ap.add_argument("--b", default=None, help="default B: a label (v1..v4) or full path")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-new", type=int,
                    default=int(INFER.get("max_new_tokens", 0) or 0))
    ap.add_argument("--max-seq-length", type=int,
                    default=int(INFER.get("max_seq_length", 4096)))
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    runs_root = Path(os.path.expanduser(args.runs_root))
    models = [m for m in MODELS
              if (runs_root / m["id"] / "adapter_config.json").exists()]
    missing = [m["label"] for m in MODELS if m not in models]
    if missing:
        print(f"NOTE: not on disk, omitted from the picker: {', '.join(missing)}")
    if not models:
        sys.exit(f"none of the configured checkpoints exist under {runs_root}")
    for m in models:
        print(f"  {m['label']} = {m['id']}")

    ids = {m["id"] for m in models}
    by_label = {m["label"]: m["id"] for m in models}
    def resolve(x, fallback):
        if not x:
            return fallback
        return by_label.get(x, x)      # accept "v3" or the full path
    default_b = resolve(args.b, models[-1]["id"])
    default_a = resolve(args.a, models[-2]["id"] if len(models) > 1 else models[0]["id"])
    for d in (default_a, default_b):
        if d not in ids:
            sys.exit(f"{d!r} is not one of: {', '.join(sorted(by_label))}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG["model"], max_seq_length=args.max_seq_length,
        dtype=torch.float16, load_in_4bit=True, revision=CFG.get("model_revision"),
    )
    # PeftModel needs one adapter at construction; the rest attach lazily.
    from peft import PeftModel
    first = default_b
    model = PeftModel.from_pretrained(
        model, str(runs_root / first), adapter_name=first.replace("/", "__"))
    FastLanguageModel.for_inference(model)

    STATE.update(
        model=model, tokenizer=tokenizer, models=models, runs_root=runs_root,
        lru=[first], max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new, seed=args.seed,
        default_a=default_a, default_b=default_b,
        labels={m["id"]: f'{m["label"]}  ({m["id"]})' for m in models},
    )
    print(f"\n  A = {default_a}\n  B = {default_b}")
    print(f"\n  open http://localhost:{args.port}\n", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
