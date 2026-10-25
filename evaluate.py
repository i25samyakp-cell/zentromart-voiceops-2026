#!/usr/bin/env python3
"""Evaluate baseline, grounded answers, retrieval, and conversations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from src.agent import ZentroMartAgent
from src.config import get_settings
from src.knowledge import KnowledgeBase
from src.llm import LLMClient
from src.utils import detect_reply_language, write_json


def split_list(value: str, separator: str = " ; ") -> list[str]:
    return [item.strip() for item in value.split(separator) if item.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def required_facts_present(reply: str, specification: str) -> tuple[bool, list[str]]:
    if not specification.strip():
        return True, []
    haystack = normalize(reply)
    missing: list[str] = []
    for fact in split_list(specification):
        alternatives = [normalize(item) for item in fact.split(" / ") if item.strip()]
        if alternatives and not any(item in haystack for item in alternatives):
            missing.append(fact)
    return not missing, missing


def forbidden_facts_absent(reply: str, specification: str) -> tuple[bool, list[str]]:
    if not specification.strip():
        return True, []
    haystack = normalize(reply)
    found = [item for item in split_list(specification) if normalize(item) in haystack]
    return not found, found


def evidence_recall(returned: list[str], gold: str) -> float | None:
    gold_ids = split_list(gold)
    if not gold_ids:
        return None
    returned_set = set(returned)
    return sum(item in returned_set for item in gold_ids) / len(gold_ids)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def retrieval_evaluation(kb: KnowledgeBase, rows: list[dict], top_k: int) -> dict:
    results = []
    recalls: list[float] = []
    answerable_count = 0
    hit_count = 0
    for row in rows:
        sources = kb.retrieve(row["question"], top_k=top_k)
        ids = [item.source_id for item in sources]
        recall = evidence_recall(ids, row["gold_evidence"])
        if recall is not None:
            answerable_count += 1
            recalls.append(recall)
            if recall > 0:
                hit_count += 1
        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "gold_evidence": split_list(row["gold_evidence"]),
                "retrieved": ids,
                "recall": recall,
            }
        )
    return {
        "suite": "retrieval",
        "top_k": top_k,
        "question_count": len(rows),
        "answerable_with_gold_count": answerable_count,
        "at_least_one_gold_hit_rate": hit_count / answerable_count if answerable_count else 0,
        "mean_gold_evidence_recall": sum(recalls) / len(recalls) if recalls else 0,
        "results": results,
    }


def question_evaluation(settings, rows: list[dict], mode: str, lexical_only: bool) -> dict:
    agent = None
    baseline = None
    if mode == "rag":
        agent = ZentroMartAgent(settings, use_embeddings=not lexical_only)
    else:
        baseline = LLMClient(settings)

    results = []
    for row in rows:
        if agent:
            agent.reset()
            output = agent.respond(row["question"])
            decision, reply = output.decision, output.reply
            source_ids = [item["source_id"] for item in output.sources]
            tool_calls = output.tool_calls
        else:
            output = baseline.baseline(row["question"], detect_reply_language(row["question"]))
            decision, reply = output.decision, output.reply
            source_ids, tool_calls = [], []

        acceptable = split_list(row.get("acceptable_decisions", row["expected_decision"]))
        decision_ok = decision in acceptable
        include_ok, missing = required_facts_present(reply, row["must_include"])
        exclude_ok, forbidden = forbidden_facts_absent(reply, row["must_not_include"])
        recall = evidence_recall(source_ids, row["gold_evidence"]) if mode == "rag" else None
        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "decision": decision,
                "acceptable_decisions": acceptable,
                "reply": reply,
                "source_ids": source_ids,
                "tool_calls": tool_calls,
                "decision_ok": decision_ok,
                "must_include_ok": include_ok,
                "missing_facts": missing,
                "must_not_include_ok": exclude_ok,
                "forbidden_facts_found": forbidden,
                "evidence_recall": recall,
                "passed": decision_ok and include_ok and exclude_ok,
            }
        )
    return summarize(results, f"questions_{mode}")


def conversation_evaluation(settings, rows: list[dict], lexical_only: bool) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["conversation_id"]].append(row)
    agent = ZentroMartAgent(settings, use_embeddings=not lexical_only)
    results = []
    for conversation_id, turns in groups.items():
        agent.reset()
        for row in sorted(turns, key=lambda item: int(item["turn"])):
            output = agent.respond(row["customer_message"])
            reply = output.reply
            include_ok, missing = required_facts_present(reply, row["must_include"])
            exclude_ok, forbidden = forbidden_facts_absent(reply, row["must_not_include"])
            called_names = [item["name"] for item in output.tool_calls]
            expected_tools = re.findall(r"(lookup_order|create_ticket)\(", row["expected_tool_calls"])
            tools_ok = all(name in called_names for name in expected_tools)
            results.append(
                {
                    "id": f"{conversation_id}-T{row['turn']}",
                    "scenario": row["scenario"],
                    "customer_message": row["customer_message"],
                    "decision": output.decision,
                    "expected_decision": row["expected_decision"],
                    "reply": reply,
                    "source_ids": [item["source_id"] for item in output.sources],
                    "tool_calls": output.tool_calls,
                    "decision_ok": output.decision == row["expected_decision"],
                    "must_include_ok": include_ok,
                    "missing_facts": missing,
                    "must_not_include_ok": exclude_ok,
                    "forbidden_facts_found": forbidden,
                    "tools_ok": tools_ok,
                    "passed": (
                        output.decision == row["expected_decision"]
                        and include_ok
                        and exclude_ok
                        and tools_ok
                    ),
                }
            )
    return summarize(results, "conversations")


def summarize(results: list[dict], suite: str) -> dict:
    count = len(results)
    passes = sum(bool(item["passed"]) for item in results)
    decision_hits = sum(bool(item["decision_ok"]) for item in results)
    recalls = [item["evidence_recall"] for item in results if item.get("evidence_recall") is not None]
    return {
        "suite": suite,
        "count": count,
        "passed": passes,
        "pass_rate": passes / count if count else 0,
        "decision_accuracy": decision_hits / count if count else 0,
        "mean_evidence_recall": sum(recalls) / len(recalls) if recalls else None,
        "results": results,
    }


def write_csv_summary(path: Path, report: dict) -> None:
    results = report["results"]
    if not results:
        return
    scalar_keys = sorted(
        {key for row in results for key, value in row.items() if not isinstance(value, (list, dict))}
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["retrieval", "baseline", "questions", "conversations", "all"],
        default="retrieval",
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation_outputs"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lexical-only", action="store_true")
    args = parser.parse_args()

    settings = get_settings(args.dataset)
    settings.validate_dataset()
    question_rows = read_rows(settings.practice_dir / "questions.csv")
    conversation_rows = read_rows(settings.practice_dir / "conversations.csv")
    args.output.mkdir(parents=True, exist_ok=True)

    suites = (
        ["retrieval", "baseline", "questions", "conversations"]
        if args.suite == "all"
        else [args.suite]
    )
    reports = []
    for suite in suites:
        if suite == "retrieval":
            report = retrieval_evaluation(
                KnowledgeBase(settings, use_embeddings=not args.lexical_only),
                question_rows,
                args.top_k,
            )
        elif suite == "baseline":
            report = question_evaluation(settings, question_rows, "baseline", args.lexical_only)
        elif suite == "questions":
            report = question_evaluation(settings, question_rows, "rag", args.lexical_only)
        else:
            report = conversation_evaluation(settings, conversation_rows, args.lexical_only)
        reports.append(report)
        json_path = args.output / f"{report['suite']}.json"
        csv_path = args.output / f"{report['suite']}.csv"
        write_json(json_path, report)
        write_csv_summary(csv_path, report)
        metric = report.get("pass_rate", report.get("mean_gold_evidence_recall", 0))
        print(f"{report['suite']}: {metric:.1%} -> {json_path}")

    write_json(args.output / "evaluation_summary.json", reports)


if __name__ == "__main__":
    main()

