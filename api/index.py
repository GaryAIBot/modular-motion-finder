import json
import os
import urllib.error
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1200)
    configuration: dict
    history: list[dict] = Field(default_factory=list)


ALLOWED = {
    "x": (1000, 30000),
    "y": (500, 12000),
    "z": (250, 3000),
    "payload": (5, 2500),
    "stiffness": (1, 3),
    "duty_cycle": (10, 100),
}


def clamp_updates(updates: dict) -> dict:
    safe = {}
    for key, value in updates.items():
        if key in ALLOWED:
            lo, hi = ALLOWED[key]
            try:
                safe[key] = max(lo, min(hi, float(value)))
            except (TypeError, ValueError):
                continue
        elif key == "environment" and value in {"clean", "industrial", "harsh"}:
            safe[key] = value
        elif key == "temperature" and value in {"standard", "cold", "hot"}:
            safe[key] = value
    return safe


@app.get("/api/health")
def health():
    return {"ok": True, "llm": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/api/chat")
def chat(body: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "The configuration assistant is not connected.")

    system = """You are an expert industrial motion-system configuration adviser inside a concept configurator.
Help the user iteratively define a modular 3-axis gantry. Be extremely concise: normally 1-2 short sentences, maximum 45 words. State that sizing and prices are indicative only when relevant.
You may update only: x, y, z in mm; payload in kg; stiffness 1 standard/2 medium/3 high; duty_cycle percent; environment clean/industrial/harsh; temperature standard/cold/hot.
CRITICAL CONFIRMATION RULE: Never change a parameter immediately after a new request. First briefly state the exact proposed changes and ask for permission; return an empty updates object. Only return parameter updates after the user explicitly confirms them (for example yes, confirm, apply, go ahead) in a later message. Use the recent conversation to identify the proposal being confirmed. If intent is unclear, ask one focused question. Never invent certifications or claim final engineering validation.
Return JSON only with exactly: {"message": string, "updates": object, "next_questions": string[]}.
When applying confirmed changes, say what changed and one key consequence. next_questions may contain up to 2 short suggestions."""
    context = {
        "current_configuration": body.configuration,
        "recent_conversation": body.history[-8:],
        "user_message": body.message,
    }
    payload = {
        "model": "gpt-5-mini",
        "instructions": system,
        "input": json.dumps(context),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "configuration_advice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "updates": {
                            "type": "object",
                            "properties": {
                                "x": {"type": ["number", "null"]},
                                "y": {"type": ["number", "null"]},
                                "z": {"type": ["number", "null"]},
                                "payload": {"type": ["number", "null"]},
                                "stiffness": {"type": ["number", "null"]},
                                "duty_cycle": {"type": ["number", "null"]},
                                "environment": {"type": ["string", "null"]},
                                "temperature": {"type": ["string", "null"]},
                            },
                            "required": ["x", "y", "z", "payload", "stiffness", "duty_cycle", "environment", "temperature"],
                            "additionalProperties": False,
                        },
                        "next_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                    },
                    "required": ["message", "updates", "next_questions"],
                    "additionalProperties": False,
                },
            }
        },
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            raw = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise HTTPException(502, f"OpenAI request failed: {detail}") from exc
    except Exception as exc:
        raise HTTPException(502, "The assistant could not complete the request.") from exc

    output_text = raw.get("output_text")
    if not output_text:
        for item in raw.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    output_text = content.get("text")
                    break
    try:
        result = json.loads(output_text or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(502, "The assistant returned an invalid configuration response.") from exc
    result["updates"] = clamp_updates({k: v for k, v in result.get("updates", {}).items() if v is not None})
    return result
