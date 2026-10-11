"""Orbit personal coach — Bedrock Converse on aggregates only (no images / screen content)."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import boto3

PROMPT_VERSION = "orbit-coach-v1"
DEFAULT_MODEL_ID = "nvidia.nemotron-nano-12b-v2"

SYSTEM_PROMPT = """You are Orbit, a private self-improvement coach for ONE person's phone habits.
You receive ONLY numeric aggregates (minutes, opens, short dips, scroll estimates, late-night use).
You NEVER receive screenshots, messages, names of contacts, or screen content.
Be warm, direct, and practical. No shame. No corporate fluff.
Respond with ONLY valid JSON (no markdown) using this schema:
{
  "pattern": "1-2 sentence observation about today's habits",
  "tips": ["tip 1", "tip 2", "tip 3"],
  "focus": "one short focus phrase for tomorrow (under 8 words)"
}
Tips must be concrete actions the user can take today or tomorrow.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")
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


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    tips_raw = raw.get("tips") or []
    if isinstance(tips_raw, str):
        tips = [tips_raw]
    else:
        tips = [str(t).strip() for t in tips_raw if str(t).strip()]
    while len(tips) < 3:
        tips.append("Protect one short focus block tomorrow.")
    return {
        "pattern": str(raw.get("pattern") or "").strip() or "Patterns are still forming — keep Orbit watching.",
        "tips": tips[:3],
        "focus": str(raw.get("focus") or "").strip() or "One intentional block",
    }


def _user_prompt(payload: dict[str, Any]) -> str:
    # Compact, privacy-safe summary for the model
    today = payload.get("today") or {}
    apps = today.get("top_apps") or []
    lines = [
        "Personal habit aggregates for coaching (this device only):",
        f"range={payload.get('range', 'today')}",
        f"total_screen_minutes={today.get('total_minutes', 0)}",
        f"unlocks={today.get('unlocks', 0)}",
        f"opens={today.get('opens', 0)}",
        f"short_dips={today.get('short_reopens', 0)}",
        f"pickups={today.get('pickups', 0)}",
        f"scroll_estimate={today.get('scroll_estimate', 0)}",
        f"late_night_minutes={today.get('late_night_minutes', 0)}",
        f"night_bursts={today.get('night_bursts', 0)}",
        f"focus_minutes={today.get('focus_minutes', 0)}",
        f"drift_minutes={today.get('drift_minutes', 0)}",
        f"limited_apps={today.get('limited_apps', [])}",
        "top_apps (label, minutes, opens, short_dips, scroll_est, tag, limited):",
    ]
    for a in apps[:8]:
        lines.append(
            f"- {a.get('label')}: {a.get('minutes')}m, opens={a.get('opens')}, "
            f"dips={a.get('short_reopens')}, scrolls~{a.get('scroll_estimate')}, "
            f"tag={a.get('tag')}, limited={a.get('limited')}"
        )
    lines.append("Write the JSON coach response now.")
    return "\n".join(lines)


def coach_from_aggregates(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = (
        os.getenv("BEDROCK_API_KEY")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "BEDROCK_API_KEY is missing. Put it in .env and restart the server."
        )

    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
    region = os.getenv("BEDROCK_REGION", "us-east-1").strip() or "us-east-1"
    model_id = (
        os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    )

    client = boto3.client("bedrock-runtime", region_name=region)
    messages = [
        {
            "role": "user",
            "content": [{"text": _user_prompt(payload)}],
        }
    ]

    started = time.perf_counter()
    response = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=messages,
        inferenceConfig={"maxTokens": 512, "temperature": 0.4},
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

    coach = _normalize(_extract_json(model_text))

    return {
        "coach": coach,
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
