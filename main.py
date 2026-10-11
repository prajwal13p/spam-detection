import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bedrock_classify import classify_image
from bedrock_insights import coach_from_aggregates

load_dotenv()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Orbit + Caller ID QA API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class InsightsRequest(BaseModel):
    """Aggregates only — never screenshots or message content."""

    range: str = "today"
    today: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health():
    has_key = bool(
        (os.getenv("BEDROCK_API_KEY") or os.getenv("AWS_BEARER_TOKEN_BEDROCK") or "").strip()
    )
    return {
        "status": "ok",
        "bedrock_key_configured": has_key,
        "model": os.getenv("BEDROCK_MODEL_ID", "nvidia.nemotron-nano-12b-v2"),
        "region": os.getenv("BEDROCK_REGION", "us-east-1"),
        "endpoints": ["/health", "/image", "/insights"],
    }


@app.post("/insights")
def insights(body: InsightsRequest):
    today = body.today or {}
    if not today:
        raise HTTPException(status_code=400, detail="today aggregates are required.")

    payload = {"range": body.range or "today", "today": today}
    try:
        analysis = coach_from_aggregates(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Bedrock coach failed: {exc}",
        ) from exc

    stamp = int(time.time())
    result = {
        "success": True,
        "pattern": analysis["coach"]["pattern"],
        "tips": analysis["coach"]["tips"],
        "focus": analysis["coach"]["focus"],
        "model": analysis["model"],
        "prompt_version": analysis["prompt_version"],
        "tokens": analysis["tokens"],
        "time_taken_ms": analysis["time_taken_ms"],
        "time_taken_seconds": analysis["time_taken_seconds"],
        "stop_reason": analysis.get("stop_reason"),
    }
    result_path = RESULTS_DIR / f"orbit_insights_{stamp}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["result_saved_as"] = str(result_path)
    return result


@app.post("/image")
async def upload_image(image: UploadFile = File(...)):
    content_type = (image.content_type or "").split(";")[0].strip().lower()
    filename = (image.filename or "").lower()
    extension_ok = filename.endswith(
        (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
    )
    if content_type not in ALLOWED_CONTENT_TYPES and not extension_ok:
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpeg, png, webp, gif, or bmp).",
        )

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file.")

    original = Path(image.filename or "capture.png").name
    stem = Path(original).stem or "capture"
    suffix = Path(original).suffix or ".png"
    stamp = int(time.time())
    dest = UPLOAD_DIR / f"{stem}_{stamp}{suffix}"
    dest.write_bytes(contents)

    try:
        analysis = classify_image(
            contents,
            content_type=content_type or "image/png",
            filename=original,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Bedrock classification failed: {exc}",
        ) from exc

    payload = {
        "success": True,
        "filename": image.filename,
        "saved_as": str(dest),
        "content_type": content_type or "image/png",
        "size_bytes": len(contents),
        "classification": analysis["classification"]["classification"],
        "phone_number": analysis["classification"]["phone_number"],
        "visible_label": analysis["classification"]["visible_label"],
        "confidence": analysis["classification"]["confidence"],
        "evidence": analysis["classification"]["evidence"],
        "model": analysis["model"],
        "prompt_version": analysis["prompt_version"],
        "tokens": analysis["tokens"],
        "time_taken_ms": analysis["time_taken_ms"],
        "time_taken_seconds": analysis["time_taken_seconds"],
        "stop_reason": analysis.get("stop_reason"),
    }

    result_path = RESULTS_DIR / f"{stem}_{stamp}.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["result_saved_as"] = str(result_path)

    return payload
