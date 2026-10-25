"""Grounded, stateful ZentroMart customer-support agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import Settings
from .knowledge import KnowledgeBase, SourceChunk
from .llm import LLMClient, ModelDecision
from .tools_layer import ToolLayer
from .utils import (
    detect_reply_language,
    extract_order_id,
    extract_verification_value,
    redact_order,
)


SYSTEM_PROMPT = """You are ZentroMart's AI customer-support assistant.

NON-NEGOTIABLE RULES
1. Use only the supplied SOURCES and verified ORDER DATA. Never invent a policy,
   date, amount, item fact, completed action, or exception.
2. Current policy outranks agent guide, help, FAQ, and marketing, in that order.
   A superseded policy applies only where the current returns policy explicitly
   preserves it.
3. If the sources do not answer the question, say you do not have that
   information and offer a human colleague. Do not guess.
4. Never reveal order information unless ORDER DATA says verified=true.
5. Never reveal a phone number, email, street address, or verification value.
6. Never say an order was cancelled, a refund approved, a replacement arranged,
   or a ticket created unless a corresponding successful tool result is supplied.
7. Use 20 October 2026 as today. Business days are Monday-Saturday excluding
   listed holidays. Return windows use calendar days.
8. Keep the customer reply to two or three short sentences. Ask only one question
   at a time. Use the customer's language: en, hi, or natural Hinglish.
9. For voice-friendly wording, say dates with month names and group order IDs in
   two groups of four digits.
10. When human review is required, choose decision=escalate and request a ticket.
    Never promise the human's outcome or an unsupported response time.

Return one JSON object only:
{
  "decision": "answer|clarify|refuse|escalate",
  "reply": "customer-facing reply",
  "evidence": ["source IDs actually used"],
  "create_ticket": true|false,
  "ticket_reason": "short machine-readable reason",
  "ticket_summary": "issue, intent, order ID if known, checks made, and reason"
}
Only cite source IDs that appear in SOURCES. Do not show source IDs to the customer.
"""


@dataclass
class AgentResult:
    reply: str
    decision: str
    sources: list[dict]
    tool_calls: list[dict]
    language: str

    def to_dict(self) -> dict:
        return {
            "reply": self.reply,
            "decision": self.decision,
            "sources": self.sources,
            "tool_calls": self.tool_calls,
            "language": self.language,
        }


@dataclass
class ConversationState:
    introduced: bool = False
    order_id: str | None = None
    verified: bool = False
    verification_failures: int = 0
    order: dict | None = None
    messages: list[dict[str, str]] = field(default_factory=list)


class ZentroMartAgent:
    def __init__(
        self,
        settings: Settings,
        use_embeddings: bool = True,
        top_k: int = 8,
    ):
        settings.validate_dataset()
        self.settings = settings
        self.kb = KnowledgeBase(settings, use_embeddings=use_embeddings)
        self.tools = ToolLayer(settings.tools_file)
        self.llm = LLMClient(settings)
        self.top_k = top_k
        self.state = ConversationState()

    def reset(self) -> None:
        self.state = ConversationState()

    @staticmethod
    def _intro(language: str) -> str:
        if language == "hi":
            return "नमस्ते, मैं ZentroMart की एआई सहायता सहायक हूँ।"
        if language == "hi-en":
            return "Namaste, main ZentroMart ki AI support assistant hoon."
        return "Hi, I'm ZentroMart's AI support assistant."

    @staticmethod
    def _ask_order_id(language: str) -> str:
        if language == "hi":
            return "कृपया अपना 8 अंकों का ऑर्डर नंबर बताइए।"
        if language == "hi-en":
            return "Please apna 8-digit order number batayein."
        return "Please share your 8-digit order number."

    @staticmethod
    def _ask_verification(language: str, retry: bool = False) -> str:
        if language == "hi":
            lead = "यह मेल नहीं हुआ। कृपया दोबारा" if retry else "पुष्टि के लिए कृपया"
            return f"{lead} रजिस्टर्ड फोन के आख़िरी 4 अंक या डिलीवरी पिनकोड बताइए।"
        if language == "hi-en":
            lead = "Yeh match nahi hua. Dobara" if retry else "Verification ke liye"
            return f"{lead} registered phone ke last 4 digits ya delivery pincode batayein."
        lead = "That did not match. Please try again with" if retry else "For verification, please provide"
        return f"{lead} the registered phone's last 4 digits or the delivery pincode."

    @staticmethod
    def _verification_refusal(language: str) -> str:
        if language == "hi":
            return "दो कोशिशों के बाद जानकारी मेल नहीं हुई, इसलिए मैं ऑर्डर की जानकारी साझा नहीं कर सकती। कृपया रजिस्टर्ड फोन से सहायता टीम से संपर्क करें।"
        if language == "hi-en":
            return "Do attempts ke baad details match nahi hui, isliye main order information share nahi kar sakti. Registered phone se support contact karein."
        return "The details did not match after two attempts, so I can't share this order's information. Please contact support from the registered phone."

    @staticmethod
    def _verified_reply(language: str) -> str:
        if language == "hi":
            return "धन्यवाद, ऑर्डर की पुष्टि हो गई है। अब मैं आपकी क्या मदद कर सकती हूँ?"
        if language == "hi-en":
            return "Thank you, order verify ho gaya hai. Ab main kis baat mein help karun?"
        return "Thank you, the order is verified. What would you like help with?"

    @staticmethod
    def _looks_like_order_help(message: str) -> bool:
        lower = message.casefold()
        phrases = [
            "help with an order", "order ke baare", "order के बारे", "अपने ऑर्डर",
            "my order", "mera order", "मेरे ऑर्डर",
        ]
        return any(item in lower for item in phrases)

    @staticmethod
    def _proof_only(message: str, proof: str) -> bool:
        lower = message.casefold()
        issue_terms = {
            "late", "refund", "cancel", "return", "damaged", "wrong", "missing",
            "ship", "deliver", "voucher", "compensation", "warranty", "pickup",
            "कैंसिल", "रिफंड", "देर", "वापस", "खराब",
        }
        return proof in "".join(ch for ch in message if ch.isdigit()) and not any(
            term in lower for term in issue_terms
        )

    def _finish_deterministic(
        self,
        reply: str,
        decision: str,
        language: str,
        source_ids: list[str],
        tool_calls: list[dict],
    ) -> AgentResult:
        if not self.state.introduced:
            reply = f"{self._intro(language)} {reply}"
            self.state.introduced = True
        self.state.messages.extend(
            [{"role": "assistant", "content": reply}]
        )
        source_lookup = {chunk.source_id: chunk for chunk in self.kb.chunks}
        sources = [source_lookup[item].public_dict() for item in source_ids if item in source_lookup]
        return AgentResult(reply, decision, sources, tool_calls, language)

    def _record_user(self, message: str) -> None:
        self.state.messages.append({"role": "user", "content": message})
        self.state.messages = self.state.messages[-12:]

    def _retrieve(self, message: str) -> list[SourceChunk]:
        context_parts = [message]
        if self.state.verified and self.state.order:
            order = self.state.order
            context_parts.extend(
                [
                    f"order status {order.get('status', '')}",
                    f"payment {order.get('payment_method', '')}",
                    f"delivery type {order.get('delivery_type', '')}",
                ]
            )
            for item in order.get("items", []) or []:
                context_parts.append(
                    f"{item.get('sku', '')} {item.get('title', '')} {item.get('return_category', '')}"
                )
        return self.kb.retrieve(" ".join(context_parts), top_k=self.top_k)

    def _model_prompt(self, message: str, language: str, sources: list[SourceChunk]) -> str:
        recent = self.state.messages[-8:]
        order = redact_order(self.state.order) if self.state.verified else None
        return (
            f"REFERENCE DATE: {self.settings.reference_date.isoformat()}\n"
            f"REPLY LANGUAGE: {language}\n"
            f"AI INTRODUCTION ALREADY GIVEN: {self.state.introduced}\n"
            f"VERIFIED ORDER DATA: {json.dumps(order, ensure_ascii=False, default=str)}\n"
            f"RECENT CONVERSATION: {json.dumps(recent, ensure_ascii=False)}\n\n"
            f"SOURCES:\n{self.kb.prompt_context(sources)}\n\n"
            f"CURRENT CUSTOMER MESSAGE:\n{message}"
        )

    def _create_handover_ticket(
        self, decision: ModelDecision, message: str, tool_calls: list[dict]
    ) -> dict:
        summary = decision.ticket_summary.strip() or (
            f"Customer issue: {message}. Intent: {decision.ticket_reason or 'human_handover'}. "
            f"Order: {self.state.order_id or 'not provided'}. Checked available ZentroMart policies; "
            "human review is required."
        )
        result = self.tools.create_ticket(
            summary=summary,
            order_id=self.state.order_id,
            reason=decision.ticket_reason or "human_handover",
            intent=decision.ticket_reason or "support_request",
        )
        tool_calls.append(
            {
                "name": "create_ticket",
                "arguments": {
                    "order_id": self.state.order_id,
                    "reason": decision.ticket_reason or "human_handover",
                    "summary": summary,
                },
                "result": result,
            }
        )
        return result

    def respond(self, message: str) -> AgentResult:
        message = str(message).strip()
        if not message:
            raise ValueError("Customer message cannot be empty.")
        language = detect_reply_language(message)
        tool_calls: list[dict] = []
        self._record_user(message)

        order_id = extract_order_id(message)
        if order_id and (order_id != self.state.order_id or not self.state.verified):
            self.state.order_id = order_id
            self.state.verified = False
            self.state.verification_failures = 0
            self.state.order = None
            result = self.tools.lookup_order(order_id)
            tool_calls.append(
                {"name": "lookup_order", "arguments": {"order_id": order_id}, "result": result}
            )
            return self._finish_deterministic(
                self._ask_verification(language), "clarify", language, ["IDV-2"], tool_calls
            )

        if self.state.order_id and not self.state.verified:
            proof = extract_verification_value(message)
            if proof:
                result = self.tools.lookup_order(self.state.order_id, proof)
                tool_calls.append(
                    {
                        "name": "lookup_order",
                        "arguments": {
                            "order_id": self.state.order_id,
                            "verification_value": "[REDACTED]",
                        },
                        "result": {key: value for key, value in result.items() if key != "delivery_pincode"},
                    }
                )
                if result.get("verified"):
                    self.state.verified = True
                    self.state.order = result
                    if self._proof_only(message, proof):
                        return self._finish_deterministic(
                            self._verified_reply(language), "answer", language, ["IDV-2"], tool_calls
                        )
                else:
                    self.state.verification_failures += 1
                    if self.state.verification_failures >= 2:
                        return self._finish_deterministic(
                            self._verification_refusal(language),
                            "refuse",
                            language,
                            ["IDV-3"],
                            tool_calls,
                        )
                    return self._finish_deterministic(
                        self._ask_verification(language, retry=True),
                        "clarify",
                        language,
                        ["IDV-3"],
                        tool_calls,
                    )

        if not self.state.order_id and self._looks_like_order_help(message):
            return self._finish_deterministic(
                self._ask_order_id(language), "clarify", language, ["STY-1", "IDV-2"], tool_calls
            )

        sources = self._retrieve(message)
        decision = self.llm.decide(
            SYSTEM_PROMPT,
            self._model_prompt(message, language, sources),
        )
        if not decision.reply:
            decision.reply = "Sorry, I don't have enough information to answer that."

        if decision.decision == "escalate" or decision.create_ticket:
            ticket = self._create_handover_ticket(decision, message, tool_calls)
            if ticket.get("created"):
                ticket_id = ticket.get("ticket_id", "")
                if ticket_id and ticket_id not in decision.reply:
                    decision.reply = (
                        f"{decision.reply.rstrip()} Ticket {ticket_id} has been created for a human colleague."
                    )
            else:
                decision.reply = (
                    "I couldn't create the handover ticket, so I won't claim it was created. "
                    "Please contact ZentroMart support directly."
                )

        if not self.state.introduced:
            decision.reply = f"{self._intro(language)} {decision.reply}"
            self.state.introduced = True
        self.state.messages.append({"role": "assistant", "content": decision.reply})

        allowed = {source.source_id: source for source in sources}
        used_ids = [item for item in decision.evidence if item in allowed]
        selected = [allowed[item] for item in used_ids] or sources
        return AgentResult(
            reply=decision.reply,
            decision=decision.decision,
            sources=[source.public_dict() for source in selected],
            tool_calls=tool_calls,
            language=language,
        )

