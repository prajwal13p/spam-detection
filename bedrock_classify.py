"""Classify caller-ID screenshots with Amazon Bedrock Converse."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import boto3

PROMPT_VERSION = "callerid-spam-v1"
DEFAULT_MODEL_ID = "nvidia.nemotron-nano-12b-v2"

SYSTEM_PROMPT = """You are a QA classifier for Android caller-ID screenshots.
Classify ONLY what is visibly shown in the image.
Do not guess spam just because a number is unknown.
Return UNKNOWN if the label is missing, obscured, or ambiguous.
Respond with ONLY valid JSON (no markdown) using this schema:
{
  "classification": "SPAM" | "NOT_SPAM" | "UNKNOWN",
  "phone_number": "digits/number visible on screen or empty string",
  "visible_label": "exact spam/safe/caller label text visible, or empty string",
  "confidence": 0.0,
  "evidence": "brief description of visible evidence"
}
"""

USER_PROMPT = (
    "Analyze this incoming-call / caller-ID screenshot. "
    "Extract the phone number if visible and decide SPAM / NOT_SPAM / UNKNOWN "
    "from labels like Spam, Suspected spam, Scam likely, Safe, etc."
)


def _media_format(content_type: str, filename: str) -> str:
    ct = (content_type or "").lower()
    name = (filename or "").lower()
    if "jpeg" in ct or "jpg" in ct or name.endswith((".jpg", ".jpeg")):
        return "jpeg"
    if "webp" in ct or name.endswith(".webp"):
        return "webp"
    if "gif" in ct or name.endswith(".gif"):
        return "gif"
    return "png"


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    # Strip optional markdown fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _normalize_result(raw: dict[str, Any]) -> dict[str, Any]:
    classification = str(raw.get("classification", "UNKNOWN")).upper().strip()
    if classification not in {"SPAM", "NOT_SPAM", "UNKNOWN"}:
        # common variants
        if "NOT" in classification and "SPAM" in classification:
            classification = "NOT_SPAM"
        elif "SPAM" in classification or "SCAM" in classification:
            classification = "SPAM"
        else:
            classification = "UNKNOWN"

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "classification": classification,
        "phone_number": str(raw.get("phone_number") or "").strip(),
        "visible_label": str(raw.get("visible_label") or "").strip(),
        "confidence": confidence,
        "evidence": str(raw.get("evidence") or "").strip(),
    }


def classify_image(
    image_bytes: bytes,
    *,
    content_type: str = "image/png",
    filename: str = "capture.png",
) -> dict[str, Any]:
    api_key = (
        os.getenv("BEDROCK_API_KEY")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "BEDROCK_API_KEY is missing. Put it in .env and restart the server."
        )

    # boto3 Bedrock API-key auth reads this env var
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

    region = os.getenv("BEDROCK_REGION", "us-east-1").strip() or "us-east-1"
    model_id = (
        os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    )

    client = boto3.client("bedrock-runtime", region_name=region)
    image_format = _media_format(content_type, filename)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": image_format,
                        "source": {"bytes": image_bytes},
                    }
                },
                {"text": USER_PROMPT},
            ],
        }
    ]

    started = time.perf_counter()
    response = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=messages,
        inferenceConfig={"maxTokens": 512, "temperature": 0.1},
    )
    elapsed_s = time.perf_counter() - started

    output_message = response.get("output", {}).get("message", {})
    text_parts = [
        block.get("text", "")
        for block in output_message.get("content", [])
        if isinstance(block, dict) and "text" in block
    ]
    model_text = "\n".join(part for part in text_parts if part).strip()

    usage = response.get("usage", {}) or {}
    input_tokens = int(usage.get("inputTokens") or 0)
    output_tokens = int(usage.get("outputTokens") or 0)
    total_tokens = int(usage.get("totalTokens") or (input_tokens + output_tokens))

    parsed = _normalize_result(_extract_json(model_text))

    return {
        "classification": parsed,
        "model": model_id,
        "prompt_version": PROMPT_VERSION,
        "region": region,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        },
        "time_taken_seconds": round(elapsed_s, 3),
        "time_taken_ms": int(elapsed_s * 1000),
        "stop_reason": response.get("stopReason"),
        "raw_model_text": model_text,
    }
