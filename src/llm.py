"""OpenAI-compatible language-model calls with strict JSON responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import Settings


@dataclass
class ModelDecision:
    decision: str
    reply: str
    evidence: list[str]
    create_ticket: bool = False
    ticket_reason: str = ""
    ticket_summary: str = ""


def _extract_json(text: str) -> dict:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.I)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}")
        if start >= 0 and end > start:
            return json.loads(clean[start : end + 1])
        raise


class LLMClient:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and add "
                "your key before running the chatbot or evaluation."
            )
        from openai import OpenAI

        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    def decide(self, system_prompt: str, user_prompt: str) -> ModelDecision:
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        data = _extract_json(content)
        decision = str(data.get("decision", "answer")).casefold()
        if decision not in {"answer", "clarify", "refuse", "escalate"}:
            decision = "answer"
        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        return ModelDecision(
            decision=decision,
            reply=str(data.get("reply", "")).strip(),
            evidence=[str(item) for item in evidence],
            create_ticket=bool(data.get("create_ticket", decision == "escalate")),
            ticket_reason=str(data.get("ticket_reason", "")).strip(),
            ticket_summary=str(data.get("ticket_summary", "")).strip(),
        )

    def baseline(self, question: str, language: str) -> ModelDecision:
        system = """You are ZentroMart's AI support assistant. This is a Step 1
baseline: you have not been given company documents. Answer the customer as
helpfully as you can. Keep the reply to two or three short sentences and use
the customer's language. Return JSON only with keys decision, reply, evidence,
create_ticket, ticket_reason, ticket_summary. decision must be answer, clarify,
refuse, or escalate. evidence must be an empty list."""
        return self.decide(system, f"Reply language: {language}\nCustomer: {question}")

