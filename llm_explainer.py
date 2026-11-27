"""
Natural-language explanation of a risk result using a LOCAL LLM (Ollama)
grounded in retrieved notes. The LLM never decides the risk level: it only
explains the deterministic model output. If Ollama is not running, a template
explanation is used instead.
"""

import json
import os
import time
import urllib.error
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("CONTEXTRA_LLM_MODEL", "phi3:3.8b")


def ollama_models(host=OLLAMA_HOST, timeout=1.5):
    """Names of locally available Ollama models, or None if Ollama is not reachable."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def retrieval_query(result):
    parts = [f["feature"].replace("_", " ") for f in result.get("top_factors", [])]
    parts.append("driver fatigue risk context")
    return " ".join(parts)


def build_prompt(result, notes):
    factors = "\n".join(f"- {f['text']}" for f in result["top_factors"]) or "- none stand out"
    probs = ", ".join(f"{k} {v * 100:.0f}%" for k, v in result["probabilities"].items())
    evidence = "\n".join(f"[{n['source']}] {n['text']}" for n in notes) or "(no notes retrieved)"
    return (
        "You explain a driver-monitoring risk result to a driver in plain language.\n"
        "Rules: use ONLY the facts below, do not invent numbers or sources, at most 4 sentences, "
        "mention the source file in square brackets when you use a note, say that this is an "
        "advisory and not a diagnosis, and do not give medical advice.\n\n"
        f"Risk level (decided by the model): {result['level']}\n"
        f"Class probabilities: {probs}\n"
        f"Main contributing factors:\n{factors}\n\n"
        f"Background notes:\n{evidence}\n\n"
        "Explanation:"
    )


def template_explanation(result):
    level = result["level"]
    factors = [f["text"] for f in result["top_factors"]]
    if not factors:
        return ("No single factor stands out. The driver state and driving context together look "
                f"{level.lower()} risk. This is an advisory, not a diagnosis.")
    joined = "; ".join(factors[:3])
    return (f"Risk is {level} mainly because {joined}. These signals matter more in combination "
            "than alone. This is an advisory from a research prototype, not a diagnosis.")


def explain(result, notes, model=None, host=OLLAMA_HOST, timeout=90):
    """Returns {'text', 'source', 'seconds'}. Falls back to a template if Ollama is unavailable."""
    start = time.time()
    model = model or DEFAULT_MODEL
    available = ollama_models(host)
    if available is None:
        return {"text": template_explanation(result), "source": "template (Ollama not running)",
                "seconds": round(time.time() - start, 2)}
    if model not in available:
        return {"text": template_explanation(result),
                "source": f"template (model {model} not found in Ollama)",
                "seconds": round(time.time() - start, 2)}
    body = json.dumps({
        "model": model, "prompt": build_prompt(result, notes), "stream": False,
        "options": {"temperature": 0.2, "num_predict": 180},
    }).encode("utf-8")
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = json.loads(r.read().decode("utf-8")).get("response", "").strip()
        if not text:
            raise ValueError("empty response")
        return {"text": text, "source": f"local LLM ({model}) + retrieved notes",
                "seconds": round(time.time() - start, 2)}
    except Exception as error:
        return {"text": template_explanation(result),
                "source": f"template (LLM error: {type(error).__name__})",
                "seconds": round(time.time() - start, 2)}
