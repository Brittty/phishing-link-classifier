"""Flask web application for the Phishing Link Classifier.

Provides a local web interface at http://127.0.0.1:5000 with two sections:
1. URL Phishing Checker - uses the existing URL classification model
2. Email Phishing Analyzer - explainable rule-based email analysis

Binds only to the loopback interface by default. No URLs are visited,
downloaded, or executed. No attachments are opened.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, render_template, request, jsonify
from markupsafe import escape
from werkzeug.utils import secure_filename

from src.config import load_config, Config
from src.model import load_pipeline
from src.predict import predict_url, load_model_for_prediction
from src.security import is_trusted_artifact, validate_url_input
from src.email_analyzer import (
    analyze_email,
    parse_email_file,
    parse_email_content,
    extract_urls_from_text,
    extract_urls_from_html,
    DEFAULT_MAX_EMAIL_SIZE,
    DEFAULT_MAX_ATTACHMENTS,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = DEFAULT_MAX_EMAIL_SIZE

config: Config = load_config()

_model_pipeline = None
_model_path = None
_model_error = None


def _load_model():
    global _model_pipeline, _model_path, _model_error
    default_model = config.paths.models_dir / "phishing_pipeline.joblib"
    if default_model.is_file() and is_trusted_artifact(default_model, trusted_dir=config.trusted_model_dir):
        try:
            _model_pipeline = load_pipeline(default_model)
            _model_path = str(default_model)
        except Exception as exc:
            _model_error = str(exc)
    else:
        _model_error = "No trusted model artifact found at models/phishing_pipeline.joblib"


_load_model()


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_available=_model_pipeline is not None,
        model_error=_model_error,
        model_path=_model_path,
    )


@app.route("/api/predict-url", methods=["POST"])
def api_predict_url():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    threshold = float(data.get("threshold", config.threshold.get("default", 0.5)))

    if not url:
        return jsonify({"error": "URL is required"}), 400

    is_valid, reason = validate_url_input(url)
    if not is_valid:
        return jsonify({"error": reason, "url": url}), 400

    if _model_pipeline is None:
        return jsonify({
            "error": "URL model not available",
            "url": escape(url),
            "url_hash": "",
            "is_valid": True,
        }), 503

    try:
        result = predict_url(_model_pipeline, url, threshold=threshold)
        result["url"] = escape(result.get("url", ""))
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc), "url": escape(url)}), 500


@app.route("/api/analyze-email", methods=["POST"])
def api_analyze_email():
    sender = ""
    reply_to = ""
    subject = ""
    body_text = ""
    body_html = ""
    attachments = []
    uploaded_filename = None

    if "file" in request.files:
        file = request.files["file"]
        if file and file.filename:
            uploaded_filename = secure_filename(file.filename)
            if not uploaded_filename.lower().endswith((".eml", ".txt")):
                return jsonify({"error": "Only .eml and .txt files are allowed"}), 400

            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            if file_size > DEFAULT_MAX_EMAIL_SIZE:
                return jsonify({"error": f"File exceeds maximum size of {DEFAULT_MAX_EMAIL_SIZE / 1_000_000:.1f} MB"}), 413

            try:
                content = file.read().decode("utf-8", errors="replace")
                parsed = parse_email_content(content)
                sender = parsed["sender"]
                reply_to = parsed["reply_to"]
                subject = parsed["subject"]
                body_text = parsed["body_text"]
                body_html = parsed["body_html"]
                attachments = parsed["attachments"]
            except Exception as exc:
                return jsonify({"error": f"Failed to parse email file: {exc}"}), 400

    if not sender:
        sender = request.form.get("sender", "")
    if not reply_to:
        reply_to = request.form.get("reply_to", "")
    if not subject:
        subject = request.form.get("subject", "")
    if not body_text:
        body_text = request.form.get("body_text", "")
    if not body_html:
        body_html = request.form.get("body_html", "")

    if request.is_json and not uploaded_filename and not any([sender, subject, body_text, body_html]):
        data = request.get_json(silent=True) or {}
        sender = data.get("sender", "")
        reply_to = data.get("reply_to", "")
        subject = data.get("subject", "")
        body_text = data.get("body_text", "")
        body_html = data.get("body_html", "")

    if not any([sender, subject, body_text, body_html]):
        return jsonify({"error": "No email content provided"}), 400

    threshold = float(request.form.get("threshold", config.threshold.get("default", 0.5)))
    model_path = str(config.paths.models_dir / "phishing_pipeline.joblib") if _model_pipeline else None

    try:
        result = analyze_email(
            sender=sender,
            reply_to=reply_to,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            attachments=attachments,
            model_path=model_path,
            config=config,
            threshold=threshold,
        )
        result["sender"] = escape(sender)
        result["reply_to"] = escape(reply_to)
        result["subject"] = escape(subject)
        result["body_text_preview"] = escape(body_text[:500] + ("..." if len(body_text) > 500 else ""))
        for url_result in result.get("urls", []):
            url_result["url"] = escape(url_result.get("url", ""))
            url_result["visible_text"] = escape(url_result.get("visible_text", ""))
        for att in result.get("attachments", []):
            att["filename"] = escape(att.get("filename", ""))
        for ind in result.get("indicators", []):
            ind["indicator"] = escape(ind.get("indicator", ""))
            ind["reason"] = escape(ind.get("reason", ""))
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/extract-urls", methods=["POST"])
def api_extract_urls():
    data = request.get_json(silent=True) or {}
    body_text = data.get("body_text", "")
    body_html = data.get("body_html", "")

    urls = []
    seen = set()
    for u in extract_urls_from_text(body_text):
        if u not in seen:
            urls.append({"url": escape(u), "visible_text": ""})
            seen.add(u)
    for u in extract_urls_from_html(body_html):
        if u["url"] not in seen:
            urls.append({"url": escape(u["url"]), "visible_text": escape(u.get("visible_text", ""))})
            seen.add(u["url"])

    return jsonify({"urls": urls})


@app.route("/api/model-status", methods=["GET"])
def api_model_status():
    return jsonify({
        "available": _model_pipeline is not None,
        "path": _model_path,
        "error": _model_error,
        "threshold": config.threshold.get("default", 0.5),
    })


def main():
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()