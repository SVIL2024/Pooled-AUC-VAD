"""probe_sources: can any pretrained checkpoint be fetched at all?

huggingface.co is unreachable from this box (DNS resolves into an unrelated
address block, connect times out) while pypi.org and github.com are fine.  That
splits the 2025-2026 training-free VAD line in two:

  * anything needing a SigLIP / DINOv2 / LLaMA / Qwen checkpoint -- normally
    hosted on HuggingFace -- is unavailable unless a mirror is reachable;
  * anything needing only PyPI packages and GitHub sources is fine.

Since the box appears to sit behind a filter that blocks HF, the standard
mirrors are probed explicitly rather than assumed.  Each is checked at the TCP
level and, where reachable, with an actual HTTPS request to a known-small path,
because a resolved host that refuses real traffic is worse than a closed port.
"""
from __future__ import annotations

import socket
import ssl
import sys
import urllib.request
import urllib.error

HOSTS = [
    ("huggingface.co", 443, "official HF"),
    ("hf-mirror.com", 443, "HF mirror (common CN)"),
    ("modelscope.cn", 443, "ModelScope (CN hub)"),
    ("www.modelscope.cn", 443, "ModelScope www"),
    ("download.pytorch.org", 443, "torchvision / timm weights"),
    ("openaipublic.azureedge.net", 443, "original OpenAI CLIP weights"),
    ("github.com", 443, "source + release assets"),
    ("objects.githubusercontent.com", 443, "GH release assets"),
    ("pypi.org", 443, "packages"),
    ("mirrors.aliyun.com", 443, "pip mirror"),
]

PATHS = [
    ("huggingface.co", "/api/models/openai/clip-vit-base-patch32"),
    ("hf-mirror.com", "/api/models/openai/clip-vit-base-patch32"),
    ("modelscope.cn", "/api/v1/models"),
    ("download.pytorch.org", "/models/resnet50-0676ba61.pth"),
    ("openaipublic.azureedge.net",
     "/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    ("github.com", "/nwpu-zxr/VadCLIP"),
]


def tcp(host, port, timeout=5):
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True, ""
    except Exception as exc:
        return False, type(exc).__name__
    finally:
        s.close()


def https(host, path, timeout=12):
    ctx = ssl.create_default_context()
    req = urllib.request.Request(f"https://{host}{path}",
                                 method="HEAD",
                                 headers={"User-Agent": "probe/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return str(r.status), r.headers.get("Content-Length", "?")
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}", "?"
    except Exception as exc:
        return type(exc).__name__, "?"


def main():
    print(f"{'host':<34}{'note':<20}{'TCP':<16}{'HTTPS(status,len)'}")
    print("-" * 92)
    results = {}
    for host, port, note in HOSTS:
        ok, err = tcp(host, port)
        tcp_s = "open" if ok else err
        path = next((p for h, p in PATHS if h == host), "/")
        st, ln = https(host, path) if ok else ("-", "-")
        results[host] = {"tcp": tcp_s, "https": st, "note": note}
        print(f"{host:<34}{note:<20}{tcp_s:<16}{st}, {ln}")

    print()
    usable = [h for h, r in results.items()
              if r["tcp"] == "open" and not r["https"].startswith(("Timeout",
                                                                   "Connect",
                                                                   "SSLError"))]
    print(f"hosts with working TCP+HTTPS: {usable}")
    ckpt_sources = [h for h in usable if h in
                    ("huggingface.co", "hf-mirror.com", "modelscope.cn",
                     "www.modelscope.cn", "download.pytorch.org",
                     "openaipublic.azureedge.net", "objects.githubusercontent.com")]
    print(f"of those, usable as CHECKPOINT sources: {ckpt_sources}")
    if not ckpt_sources:
        print("\nVERDICT: no pretrained-checkpoint source is reachable.")
        print("        Only PyPI/GitHub SOURCE is available, so every")
        print("        backbone and LLM must already exist locally.")
    elif "huggingface.co" not in ckpt_sources and ckpt_sources:
        print("\nVERDICT: HF blocked but mirrors work -- checkpoints are")
        print("        fetchable after switching HF_ENDPOINT.")
    else:
        print("\nVERDICT: HF reachable -- full frontier is available.")


if __name__ == "__main__":
    sys.exit(main())