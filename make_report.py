#!/usr/bin/env python3
"""Generate the four-page capstone report in first person."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#0B1F3A")
BLUE = colors.HexColor("#155EEF")
LIGHT_BLUE = colors.HexColor("#EAF2FF")
CYAN = colors.HexColor("#00A6A6")
TEXT = colors.HexColor("#243247")
MUTED = colors.HexColor("#617086")
LIGHT = colors.HexColor("#F5F7FA")
AMBER = colors.HexColor("#FFF4D6")
GREEN = colors.HexColor("#E8F7F0")


def load_json(directory: Path, names: list[str]) -> dict | None:
    for name in names:
        path = directory / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def percent(value) -> str:
    if value is None:
        return "Not run"
    return f"{float(value) * 100:.1f}%"


def metric(report: dict | None, key: str) -> float | None:
    return report.get(key) if report else None


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=33,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverSub",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=MUTED,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "H1Navy",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "H2Blue",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=BLUE,
            spaceBefore=6,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.2,
            textColor=TEXT,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallMuted",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            "Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=NAVY,
        )
    )
    return styles


def P(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str, styles) -> Paragraph:
    return P(f"- {text}", styles["BodySmall"])


def callout(text: str, styles, background=LIGHT_BLUE):
    table = Table([[P(text, styles["Callout"])]], colWidths=[167 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def architecture_table(styles):
    labels = ["Customer\naudio", "Gnani\nSTT", "RAG + LLM\n+ tools", "Gnani\nTTS", "Spoken\nreply"]
    data = [[P(label.replace("\n", "<br/>"), styles["Callout"]) for label in labels]]
    table = Table(data, colWidths=[31 * mm] * 5, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
                ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#DDE9FF")),
                ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.7, BLUE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def results_table(styles, baseline, grounded, retrieval, conversations):
    rows = [
        ["Evaluation", "Result", "What it measures"],
        ["Step 1 baseline", percent(metric(baseline, "pass_rate")), "LLM without company documents"],
        ["Step 2 grounded", percent(metric(grounded, "pass_rate")), "Decision and required-fact checks"],
        [
            "Retrieval top 8",
            percent(metric(retrieval, "mean_gold_evidence_recall")),
            "Mean gold-evidence recall",
        ],
        [
            "Retrieval hit rate",
            percent(metric(retrieval, "at_least_one_gold_hit_rate")),
            "At least one gold source found",
        ],
        ["Step 3 conversations", percent(metric(conversations, "pass_rate")), "96 turns across 23 conversations"],
    ]
    table = Table(rows, colWidths=[50 * mm, 30 * mm, 87 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("LEADING", (0, 0), (-1, -1), 10.5),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CAD3E0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def page_decor(canvas, document, participant):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 10 * mm, width, 10 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(18 * mm, height - 6.5 * mm, "VOICEOPS 2026 - CAPSTONE REPORT")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, 10 * mm, participant)
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report(output: Path, evaluation_dir: Path, participant: str, repo_url: str) -> None:
    baseline = load_json(evaluation_dir, ["questions_baseline.json"])
    grounded = load_json(evaluation_dir, ["questions_rag.json"])
    retrieval = load_json(
        evaluation_dir, ["retrieval.json", "retrieval_evaluation.json"]
    )
    conversations = load_json(evaluation_dir, ["conversations.json"])
    styles = make_styles()
    displayed_repo = (
        "Not supplied - replace before submission"
        if "REPLACE_WITH" in repo_url
        else repo_url
    )
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=21 * mm,
        leftMargin=21 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="VoiceOps 2026 Capstone Report",
        author=participant,
    )
    story = []

    # Page 1
    story.extend(
        [
            Spacer(1, 12 * mm),
            P("VoiceOps 2026", styles["CoverTitle"]),
            P("A grounded, conversational and voice-enabled support agent for ZentroMart", styles["CoverSub"]),
            Spacer(1, 2 * mm),
            Table(
                [
                    [P("Participant", styles["SmallMuted"]), P(participant, styles["BodySmall"])],
                    [P("Repository", styles["SmallMuted"]), P(displayed_repo, styles["BodySmall"])],
                    [P("Reference date", styles["SmallMuted"]), P("20 October 2026", styles["BodySmall"])],
                ],
                colWidths=[35 * mm, 132 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7DEE8")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            ),
            Spacer(1, 7 * mm),
            P("1. Project overview", styles["H1Navy"]),
            P(
                "I built one support agent in four stages: a basic LLM baseline, document-grounded retrieval, stateful conversations with verified tools, and an end-to-end Gnani voice layer. My main design goal was correctness: the agent should locate the current policy, admit when the knowledge base is silent, and never claim an action that the code did not complete.",
                styles["BodySmall"],
            ),
            Spacer(1, 3 * mm),
            architecture_table(styles),
            Spacer(1, 6 * mm),
            P("Dataset and constraints", styles["H2Blue"]),
            P(
                "I used the supplied synthetic dataset: 21 knowledge documents, seven operational CSV tables, 52 practice questions, and 23 conversations containing 96 turns. I preserve IDs as strings and use 20 October 2026 for all time calculations. Gnani is the only speech service used.",
                styles["BodySmall"],
            ),
            callout(
                "Core principle: retrieve evidence first, expose order details only after verification, then answer briefly in the customer's language.",
                styles,
            ),
        ]
    )

    # Page 2
    story.extend(
        [
            PageBreak(),
            P("2. From baseline to grounded answers", styles["H1Navy"]),
            P("Step 1 - baseline", styles["H2Blue"]),
            P(
                "I first call the LLM with only ZentroMart's role and response rules. This deliberately measures what happens before the model sees company documents. The baseline can sound confident while guessing company-specific windows, fees, and timelines, which makes it a useful comparison rather than a safe production agent.",
                styles["BodySmall"],
            ),
            P("Step 2 - hybrid RAG", styles["H2Blue"]),
            P(
                "I split Markdown files at each section ID and retain document type, status, version date, title, and authority. I index product and pincode rows as structured chunks. Retrieval combines a local BM25-style score with optional embeddings, boosts exact SKUs, limits catalog crowding, and always exposes the policy-precedence rule to the answer model.",
                styles["BodySmall"],
            ),
            results_table(styles, baseline, grounded, retrieval, conversations),
            Spacer(1, 5 * mm),
            callout(
                "API-dependent results are shown as 'Not run' until I execute evaluate.py with my own OpenAI key. I do not invent evaluation scores. The offline retrieval figures are reproducible without credentials.",
                styles,
                AMBER if baseline is None or grounded is None else GREEN,
            ),
            Spacer(1, 5 * mm),
            P("Evaluation method", styles["H2Blue"]),
            bullet("I compare each decision with the allowed decision labels.", styles),
            bullet("I check required facts and reject forbidden claims.", styles),
            bullet("I calculate gold-evidence recall from preserved source IDs.", styles),
            bullet("Practice reference answers are never provided to the runtime agent.", styles),
        ]
    )

    # Page 3
    story.extend(
        [
            PageBreak(),
            P("3. Conversation, tools and voice", styles["H1Navy"]),
            P("Conversation state", styles["H2Blue"]),
            P(
                "I keep the current order ID, verification status, verified order data, failure count, and recent messages inside one conversation. This lets the agent understand follow-ups such as 'How long will that take?' and return to an earlier order after a general-policy question.",
                styles["BodySmall"],
            ),
            P("Identity boundary", styles["H2Blue"]),
            bullet("I request the order ID before any specific lookup.", styles),
            bullet("The first lookup is unverified and cannot return order facts.", styles),
            bullet("I ask for one proof: phone last four digits or delivery pincode.", styles),
            bullet("I allow one retry, then refuse to share private order details.", styles),
            bullet("I redact verification values and sensitive fields from traces and prompts.", styles),
            P("Order and handover tools", styles["H2Blue"]),
            P(
                "I load the supplied lookup_order and create_ticket implementations directly from the dataset. A handover is confirmed only after create_ticket returns created=True. Ticket summaries contain the issue, intent, order ID when known, checks performed, and the escalation reason, without verification secrets.",
                styles["BodySmall"],
            ),
            P("Gnani voice layer", styles["H2Blue"]),
            P(
                "Customer audio is transcribed through Gnani STT, passed to the same Step 3 agent, and synthesized through Gnani TTS. My default settings are en-IN, Kaveri, and timbre-v2.5. The batch runner accepts a folder and writes a transcript, text reply, WAV reply, retrieved sources, tool trace, and machine-readable result for every clip.",
                styles["BodySmall"],
            ),
            Table(
                [
                    ["Failure", "Recovery"],
                    ["Empty transcript", "Ask the customer to repeat; do not call the LLM"],
                    ["No policy evidence", "Say I do not have the information; offer a colleague"],
                    ["Verification fails twice", "Refuse order details and suggest registered-channel support"],
                    ["Ticket tool fails", "State that no ticket was created"],
                    ["Unsupported request", "Decline politely without promising an exception"],
                ],
                colWidths=[48 * mm, 119 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CAD3E0")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            ),
        ]
    )

    # Page 4
    story.extend(
        [
            PageBreak(),
            P("4. Limitations, reproducibility and next steps", styles["H1Navy"]),
            P("Current limitations", styles["H2Blue"]),
            bullet("Answer quality still depends on the selected LLM and its structured-output reliability.", styles),
            bullet("Embedding creation and voice testing require personal API credentials that are never stored in the repository.", styles),
            bullet("The supplied verification check is an educational mock and is not production authentication.", styles),
            bullet("The dataset contains text practice cases, so I must record my own consenting voice for the final demo.", styles),
            P("What I learned", styles["H2Blue"]),
            P(
                "The strongest retrieval improvement came from respecting document authority and preventing several similar product rows from pushing policy sections out of the top results. Exact SKU matching is useful, but the product row must be paired with the governing policy section.",
                styles["BodySmall"],
            ),
            P("What I would improve", styles["H2Blue"]),
            P(
                "With more time, I would add a learned multilingual reranker, stream STT and TTS for lower latency, run adversarial privacy tests, and add observability for retrieval confidence, latency, API failures, and tool outcomes. I would also replace the educational four-digit check with production-grade authentication before handling real customer data.",
                styles["BodySmall"],
            ),
            P("Reproducibility", styles["H2Blue"]),
            P(
                "The repository provides one environment template, pinned dependency ranges, an offline validator, unit tests, a text chat, a batch voice command, and evaluation scripts. Each voice result saves the exact response, evidence IDs, and tool trace. Secrets, the dataset, virtual environments, and generated ticket files are excluded from Git.",
                styles["BodySmall"],
            ),
            Table(
                [
                    ["Command", "Purpose"],
                    [P("python validate_submission.py", styles["SmallMuted"]), P("Validate dataset, privacy gate, tests, and required files", styles["SmallMuted"])],
                    [P("python evaluate.py --suite all", styles["SmallMuted"]), P("Produce Step 1, Step 2, retrieval, and conversation results", styles["SmallMuted"])],
                    [P("python run.py --input-audio input_audio --output outputs", styles["SmallMuted"]), P("Run the required end-to-end audio batch", styles["SmallMuted"])],
                    [P("python make_report.py --evaluation-dir evaluation_outputs", styles["SmallMuted"]), P("Regenerate this report with real scores", styles["SmallMuted"])],
                ],
                colWidths=[83 * mm, 84 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTNAME", (0, 1), (0, -1), "Courier"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CAD3E0")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            ),
            Spacer(1, 7 * mm),
            callout(
                "Conclusion: I built a support agent that combines grounded evidence, conversation memory, privacy-aware tools, and Gnani speech in one reproducible pipeline. Its most important behavior is knowing when it may answer, when it must ask, and when it must hand over.",
                styles,
                GREEN,
            ),
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(
        story,
        onFirstPage=lambda c, d: page_decor(c, d, participant),
        onLaterPages=lambda c, d: page_decor(c, d, participant),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("report.pdf"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("evaluation_outputs"))
    parser.add_argument("--participant", default=os.getenv("PARTICIPANT_NAME", "Ishaan Agrawal"))
    parser.add_argument("--repo-url", default=os.getenv("PUBLIC_REPO_URL", "REPLACE_WITH_PUBLIC_GITHUB_URL"))
    args = parser.parse_args()
    build_report(args.output, args.evaluation_dir, args.participant, args.repo_url)
    print(f"Created {args.output.resolve()}")


if __name__ == "__main__":
    main()

