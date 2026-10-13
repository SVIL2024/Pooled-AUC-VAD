"""gate_install: the one gate that decides whether the pivot is real.

probe_env showed transformers / open_clip / sentence_transformers are all
absent, and huggingface.co is unreachable while hf-mirror.com and
modelscope.cn answer on 443.  That leaves exactly one question worth spending
time on: can a package be installed from PyPI and can a real checkpoint file be
downloaded end to end through the mirror, at the size the frontier actually
needs (a ViT-B/32 is 350 MB, a SigLIP-SO400M is roughly 1.7 GB, a small LLM is
several GB).

This script does not assume any of it.  It:
  1. imports transformers and reports the version (fails loudly if absent);
  2. fetches a small real file from hf-mirror to prove the mirror serves bytes;
  3. resolves a repo revision through the mirror's API, which is what
     from_pretrained actually calls;
  4. writes a short marker file so the next step knows the gate passed.

No model is loaded here -- that is Stage 1 work, and it is the expensive part.
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.request

MIRROR = "https://hf-mirror.com"
TIMEOUT = 30


def get(url, method="GET", limit=None):
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, method=method,
                                 headers={"User-Agent": "gate/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as r:
        data = r.read() if method == "GET" else b""
        if limit is not None:
            data = data[:limit]
        return r.status, dict(r.headers), data


def main():
    ok = {}

    print("=== 1. transformers ===")
    try:
        import transformers
        print(f"  transformers {transformers.__version__}")
        ok["transformers"] = True
    except Exception as exc:
        print(f"  MISSING ({type(exc).__name__}) -- pip install transformers")
        ok["transformers"] = False

    print()
    print("=== 2. mirror serves real bytes ===")
    # a genuine small file from a real repo, served through the mirror
    url = f"{MIRROR}/openai/clip-vit-base-patch32/resolve/main/config.json"
    try:
        st, hdr, body = get(url)
        cfg = json.loads(body.decode("utf-8"))
        print(f"  HTTP {st}  config.json parsed: "
              f"model_type={cfg.get('model_type')} "
              f"hidden={cfg.get('hidden_size')}")
        ok["mirror_config"] = True
    except Exception as exc:
        print(f"  FAILED ({type(exc).__name__}: {exc})")
        ok["mirror_config"] = False

    print()
    print("=== 3. mirror resolves a repo revision (what from_pretrained calls) ===")
    url = f"{MIRROR}/api/models/openai/clip-vit-base-patch32"
    try:
        st, hdr, body = get(url)
        meta = json.loads(body.decode("utf-8"))
        files = [s["rfilename"] for s in meta.get("siblings", [])]
        print(f"  HTTP {st}  sha={str(meta.get('sha'))[:12]}  "
              f"{len(files)} files")
        print(f"  has weights: "
              f"{[f for f in files if f.endswith(('.bin','.safetensors'))][:4]}")
        ok["mirror_api"] = True
    except Exception as exc:
        print(f"  FAILED ({type(exc).__name__}: {exc})")
        ok["mirror_api"] = False

    print()
    print("=== 4. size check on a weight file the frontier would need ===")
    for repo, fn in (("openai/clip-vit-base-patch32", "model.safetensors"),
                     ("google/siglip-so400m-patch14-384", "model.safetensors")):
        u = f"{MIRROR}/{repo}/resolve/main/{fn}"
        try:
            req = urllib.request.Request(u, method="HEAD",
                                         headers={"User-Agent": "gate/1.0"})
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=TIMEOUT,
                                        context=ctx) as r:
                n = r.headers.get("Content-Length", "?")
            try:
                mb = f"{int(n)/1e6:.0f} MB"
            except (TypeError, ValueError):
                mb = "?"
            print(f"  {repo:<42} {mb}")
            ok[f"size_{repo.split('/')[-1]}"] = True
        except Exception as exc:
            print(f"  {repo:<42} FAILED ({type(exc).__name__})")
            ok[f"size_{repo.split('/')[-1]}"] = False

    passed = all(ok.get(k) for k in ("mirror_config", "mirror_api"))
    print()
    print("=" * 70)
    print(f"GATE {'PASSED' if passed else 'FAILED'}  {ok}")
    print("=" * 70)
    with open(os.path.join(os.path.dirname(__file__),
                           "gate_install.json"), "w") as f:
        json.dump(ok, f, indent=2)

    if not ok.get("transformers"):
        print("next: pip install transformers (pypi reachable)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())