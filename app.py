"""
CIM Financial Intelligence — Flask UI Server
Wraps cim_extractor.py for client-facing use at Atar Capital.
All extraction logic lives in cim_extractor.py — this file is server + UI routing only.
"""

import os
import sys
import json
import uuid
import logging
import tempfile
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, make_response

# ── Bootstrap path so cim_extractor is importable ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cim_extractor import CIMParser, LLMExtractor, _load_env

_load_env()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

# ── In-memory job store  {job_id: {"status": ..., "log": [...], "result": ...}} ──
jobs: dict = {}
jobs_lock = threading.Lock()

# ── Provider / model mapping ──────────────────────────────────────────────────
PROVIDER_ENV = {
    "nvidia":    "NVIDIA_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "ollama":    "OLLAMA_API_KEY",
}

# ── Full --req string pulled directly from cim_extractor's argparse default ───
def _get_default_req() -> str:
    import argparse, inspect, re as _re, cim_extractor as _ce
    src = inspect.getsource(_ce.main)
    m = _re.search(r'add_argument\(\s*"--req".*?default=\((.*?)\),\s*help=', src, _re.DOTALL)
    if m:
        try:
            return eval(m.group(1))
        except Exception:
            pass
    return "Extract all financial fields as per system rules."

DEFAULT_REQ = _get_default_req()


# ── Background extraction worker ──────────────────────────────────────────────
class _JobLogHandler(logging.Handler):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def emit(self, record):
        msg = self.format(record)
        with jobs_lock:
            if self.job_id in jobs:
                jobs[self.job_id]["log"].append(msg)


def _run_extraction(job_id: str, pdf_path: str, provider: str, model: str, api_key: str):
    handler = _JobLogHandler(job_id)
    handler.setFormatter(logging.Formatter("%(levelname)s — %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    def _update(status, **kwargs):
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = status
                jobs[job_id].update(kwargs)

    try:
        _update("parsing")
        cim = CIMParser(pdf_path)
        cim.parse_pdf()

        _update("filtering")
        relevant = cim.find_financial_sections()
        if not relevant:
            relevant = list(cim.extracted_tables)

        _update("extracting")
        extractor = LLMExtractor(api_key=api_key, provider=provider, model_name=model)
        raw = extractor.extract_fields(relevant, client_requirements=DEFAULT_REQ)
        if raw is None:
            raise RuntimeError("LLM returned no data. Check API key and model.")

        # Save raw output
        out_dir = Path("extracted_results")
        out_dir.mkdir(exist_ok=True)
        base = Path(pdf_path).stem.replace(" ", "_")
        raw_path = str(out_dir / f"{base}_extracted.json")
        Path(raw_path).write_text(json.dumps(raw, indent=4))

        _update("done", raw=raw, raw_path=raw_path)

    except Exception as exc:
        logging.error(f"Extraction error: {exc}")
        _update("error", error=str(exc))
    finally:
        root_logger.removeHandler(handler)
        try:
            os.unlink(pdf_path)
        except OSError:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/extract", methods=["POST"])
def api_extract():
    """Start an async extraction job. Returns job_id immediately."""
    pdf_file = request.files.get("pdf")
    provider = request.form.get("provider", "nvidia")
    model    = request.form.get("model", "meta/llama-3.3-70b-instruct")
    api_key  = request.form.get("api_key", "").strip()

    if not pdf_file or not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file."}), 400
    if not api_key:
        env_var = PROVIDER_ENV.get(provider, "API_KEY")
        api_key = os.getenv(env_var, "")
    if not api_key:
        return jsonify({"error": f"API key missing for provider '{provider}'."}), 400

    # Save PDF to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_file.save(tmp.name)
    tmp.close()

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "log": [], "raw": None, "error": None}

    thread = threading.Thread(
        target=_run_extraction,
        args=(job_id, tmp.name, provider, model, api_key),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id}), 202


@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status":     job["status"],
        "log":        job["log"][-20:],
        "error":      job.get("error"),
        "has_result": job["status"] == "done",
    })


@app.route("/api/result/<job_id>")
def api_result(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Job not finished yet"}), 400
    return jsonify({"raw": job["raw"]})


@app.route("/api/download/<job_id>")
def api_download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Result not ready"}), 400

    path = job.get("raw_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)

    import io
    return send_file(
        io.BytesIO(json.dumps(job["raw"], indent=4).encode()),
        mimetype="application/json",
        as_attachment=True,
        download_name="extracted.json",
    )


@app.route("/api/env-keys")
def api_env_keys():
    result = {}
    for provider, env_var in PROVIDER_ENV.items():
        result[provider] = bool(os.getenv(env_var, ""))
    return jsonify(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    print("\n  CIM Financial Intelligence")
    print("  Running at: http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
