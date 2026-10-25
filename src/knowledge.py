"""Knowledge-base parsing and hybrid retrieval.

The index keeps the dataset's section IDs and document metadata. It can use
OpenAI embeddings when a key is available and always keeps a local BM25-style
lexical index as a deterministic fallback.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .config import PROJECT_ROOT, Settings
from .utils import cosine_similarity, extract_skus, tokenize, write_json


SECTION_RE = re.compile(r"^##\s+\[([^\]]+)]\s*(.*)$")

AUTHORITY = {
    "policy": 1.00,
    "agent_guide": 0.97,
    "help": 0.93,
    "faq": 0.88,
    "marketing": 0.76,
    "meta": 0.98,
    "catalog": 0.96,
    "serviceability": 0.96,
}

QUERY_EXPANSIONS = {
    "send back": "return returns window",
    "take back": "return returns window",
    "collection": "reverse pickup fee",
    "nobody came": "missed reverse pickup",
    "money back": "refund timeline initiation",
    "cash on delivery": "cod refund",
    "not home": "delivery attempt courier",
    "late": "promised date delayed delivery compensation",
    "crack": "damaged defective item",
    "broken": "damaged defective item",
    "wrong item": "damaged wrong missing item",
    "doesn't fit": "return eligibility innerwear fashion",
    "doesnt fit": "return eligibility innerwear fashion",
    "warranty": "brand warranty assured",
    "assured": "warranty claim My Orders 12 months",
    "claim kaise": "warranty claim My Orders assured",
    "human": "handover escalation ticket",
    "person": "handover escalation ticket",
    "cancel": "cancellation shipped unshipped",
    "cancellation": "cancel shipped unshipped",
    "address": "change delivery address shipping",
    "refund": "refund initiated payment business days",
    "never placed": "unauthorised unauthorized account activity escalation human handover",
    "did not make": "unauthorised unauthorized account activity escalation human handover",
    "crack": "damaged item high value Priority Resolution escalation",
    "worth": "product price high value escalation",
    "how long do i get": "return window category delivery date festive sale",
    "how many days": "return window category delivery date",
    "कितने दिन": "return window category delivery date",
    "लौट": "return window condition category",
    "वापस": "return window condition category",
    "खोल": "opened seal intact beauty return condition",
    "सील": "unopened seal intact beauty return condition",
    "रिफंड": "refund initiated payment business days",
    "कैंसिल": "cancellation shipped refuse parcel refund",
    "खराब": "damaged defective report 48 hours photos",
    "गलत": "wrong item report 48 hours photos",
}

SECTION_HINTS = {
    "return": {"RET-2", "RET-3", "RET-4", "RET-5", "RET-6", "RET-8"},
    "refund": {"REF-1", "REF-2", "REF-3", "REF-4", "KB-00.3"},
    "warranty": {"WAR-1", "WAR-2", "MKS-3"},
    "assured": {"WAR-2", "MKS-3"},
    "cancel": {"CAN-1", "CAN-2", "CAN-3", "REF-1"},
    "cancellation": {"CAN-1", "CAN-2", "CAN-3", "REF-1"},
    "delivery": {"DEL-1", "DEL-2", "DEL-3", "DEL-4", "DEL-5", "DEL-6"},
    "courier": {"DEL-3", "TRK-1"},
    "damaged": {"DWM-1", "DWM-2", "DWM-3", "DWM-5", "ESC-1"},
    "crack": {"DWM-1", "DWM-3", "ESC-1"},
    "human": {"ESC-1", "ESC-2", "ESC-3"},
    "never placed": {"ESC-1"},
    "speaker": {"TRB-4"},
    "coupon": {"CPN-1", "CPN-2", "REF-4"},
    "लौट": {"RET-2", "RET-4", "RET-5", "RET-8"},
    "वापस": {"RET-2", "RET-4", "RET-5"},
    "खोल": {"RET-4"},
    "रिफंड": {"REF-1", "REF-2", "REF-3", "KB-00.3"},
    "कैंसिल": {"FAQH-1", "CAN-1", "CAN-2", "REF-1"},
    "खराब": {"FAQH-3", "DWM-1", "DWM-2", "DWM-3"},
}


@dataclass
class SourceChunk:
    source_id: str
    title: str
    text: str
    file_name: str
    doc_id: str
    doc_type: str
    status: str
    effective_from: str
    language: str = "en"
    score: float = 0.0

    def as_prompt_text(self) -> str:
        return (
            f"SOURCE {self.source_id} | {self.title} | type={self.doc_type} "
            f"| status={self.status} | effective_from={self.effective_from}\n"
            f"{self.text}"
        )

    def public_dict(self) -> dict:
        data = asdict(self)
        data["score"] = round(self.score, 6)
        return data


def _parse_frontmatter(lines: list[str]) -> tuple[dict[str, str], int]:
    if not lines or lines[0].strip() != "---":
        return {}, 0
    metadata: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return metadata, index + 1
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata, 0


def parse_markdown(path: Path) -> list[SourceChunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata, start = _parse_frontmatter(lines)
    title = metadata.get("title", path.stem)
    chunks: list[SourceChunk] = []
    current_id: str | None = None
    current_title = ""
    body: list[str] = []

    def finish() -> None:
        if not current_id:
            return
        clean = "\n".join(body).strip()
        if clean:
            chunks.append(
                SourceChunk(
                    source_id=current_id,
                    title=f"{title} - {current_title}".strip(" -"),
                    text=clean,
                    file_name=path.name,
                    doc_id=metadata.get("doc_id", path.stem),
                    doc_type=metadata.get("doc_type", "unknown"),
                    status=metadata.get("status", "current"),
                    effective_from=metadata.get("effective_from", ""),
                    language=metadata.get("language", "en"),
                )
            )

    for line in lines[start:]:
        match = SECTION_RE.match(line)
        if match:
            finish()
            current_id, current_title = match.group(1).strip(), match.group(2).strip()
            body = []
        elif current_id:
            body.append(line)
    finish()
    return chunks


def _read_catalog(dataset_dir: Path) -> list[SourceChunk]:
    path = dataset_dir / "data" / "products.csv"
    chunks: list[SourceChunk] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sku = row["sku"]
            text = "; ".join(f"{key}={value}" for key, value in row.items())
            chunks.append(
                SourceChunk(
                    source_id=f"products.csv:{sku}",
                    title=f"Product catalog - {row['title']}",
                    text=text,
                    file_name="products.csv",
                    doc_id=sku,
                    doc_type="catalog",
                    status="current",
                    effective_from="2026-10-20",
                )
            )
    return chunks


def _read_pincodes(dataset_dir: Path) -> list[SourceChunk]:
    path = dataset_dir / "data" / "pincodes.csv"
    chunks: list[SourceChunk] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pin = row["pincode"]
            text = "; ".join(f"{key}={value}" for key, value in row.items())
            chunks.append(
                SourceChunk(
                    source_id=f"pincodes.csv:{pin}",
                    title=f"Pincode serviceability - {pin}",
                    text=text,
                    file_name="pincodes.csv",
                    doc_id=pin,
                    doc_type="serviceability",
                    status="current",
                    effective_from="2026-10-20",
                )
            )
    return chunks


class KnowledgeBase:
    def __init__(self, settings: Settings, use_embeddings: bool = True):
        self.settings = settings
        self.use_embeddings = bool(use_embeddings and settings.openai_api_key)
        self.chunks = self._load_chunks()
        self._doc_tokens = [tokenize(self._search_text(chunk)) for chunk in self.chunks]
        self._doc_freq = self._build_doc_freq(self._doc_tokens)
        self._avg_len = sum(map(len, self._doc_tokens)) / max(1, len(self._doc_tokens))
        self._embeddings: list[list[float]] | None = None
        if self.use_embeddings:
            self._embeddings = self._load_or_create_embeddings()

    def _load_chunks(self) -> list[SourceChunk]:
        chunks: list[SourceChunk] = []
        for path in sorted(self.settings.kb_dir.glob("*.md")):
            chunks.extend(parse_markdown(path))
        chunks.extend(_read_catalog(self.settings.dataset_dir))
        chunks.extend(_read_pincodes(self.settings.dataset_dir))
        return chunks

    @staticmethod
    def _search_text(chunk: SourceChunk) -> str:
        return " ".join(
            [chunk.source_id, chunk.title, chunk.text, chunk.doc_type, chunk.status]
        )

    @staticmethod
    def _build_doc_freq(doc_tokens: Iterable[list[str]]) -> Counter:
        frequencies: Counter = Counter()
        for tokens in doc_tokens:
            frequencies.update(set(tokens))
        return frequencies

    def _expanded_query(self, query: str) -> str:
        lowered = query.casefold()
        additions = [value for key, value in QUERY_EXPANSIONS.items() if key in lowered]
        skus = extract_skus(query)
        return " ".join([query, *additions, *skus])

    def _bm25_scores(self, query: str) -> list[float]:
        terms = tokenize(self._expanded_query(query))
        n_docs = len(self._doc_tokens)
        scores: list[float] = []
        for chunk, tokens in zip(self.chunks, self._doc_tokens):
            tf = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in terms:
                frequency = tf[term]
                if not frequency:
                    continue
                df = self._doc_freq[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * length / max(self._avg_len, 1)
                )
                score += idf * frequency * 2.5 / denominator

            lowered = query.casefold()
            if chunk.source_id.casefold() in lowered:
                score += 15.0
            if chunk.doc_id.casefold() in lowered:
                score += 12.0
            if chunk.source_id.startswith("products.csv:"):
                sku = chunk.source_id.split(":", 1)[1]
                if sku in extract_skus(query):
                    score += 25.0
                product_title = chunk.title.split(" - ", 1)[-1].casefold()
                title_terms = set(tokenize(product_title))
                overlap = len(title_terms & set(terms))
                if overlap >= 2:
                    score += 3.0 * overlap
            if chunk.source_id.startswith("pincodes.csv:") and chunk.doc_id in query:
                score += 25.0
            for phrase, section_ids in SECTION_HINTS.items():
                if phrase in lowered and chunk.source_id in section_ids:
                    score += 5.0
            scores.append(score)
        return scores

    def _cache_signature(self) -> str:
        content = "\n".join(
            f"{chunk.source_id}|{chunk.status}|{chunk.effective_from}|{chunk.text}"
            for chunk in self.chunks
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _load_or_create_embeddings(self) -> list[list[float]]:
        cache_path = PROJECT_ROOT / ".cache" / "rag_index.json"
        signature = self._cache_signature()
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("signature") == signature
                and cached.get("model") == self.settings.embedding_model
                and len(cached.get("embeddings", [])) == len(self.chunks)
            ):
                return cached["embeddings"]

        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)
        texts = [self._search_text(chunk)[:8000] for chunk in self.chunks]
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), 64):
            response = client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts[start : start + 64],
            )
            embeddings.extend(item.embedding for item in response.data)
        write_json(
            cache_path,
            {
                "signature": signature,
                "model": self.settings.embedding_model,
                "source_ids": [chunk.source_id for chunk in self.chunks],
                "embeddings": embeddings,
            },
        )
        return embeddings

    def _query_embedding(self, query: str) -> list[float] | None:
        if not self._embeddings:
            return None
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.embeddings.create(
            model=self.settings.embedding_model,
            input=self._expanded_query(query),
        )
        return response.data[0].embedding

    def retrieve(self, query: str, top_k: int = 8) -> list[SourceChunk]:
        lexical = self._bm25_scores(query)
        max_lexical = max(lexical) if lexical else 1.0
        query_vector = self._query_embedding(query)
        scored: list[SourceChunk] = []

        for index, chunk in enumerate(self.chunks):
            lexical_score = lexical[index] / max(max_lexical, 1e-9)
            dense_score = (
                cosine_similarity(query_vector, self._embeddings[index])
                if query_vector is not None and self._embeddings is not None
                else 0.0
            )
            authority = AUTHORITY.get(chunk.doc_type, 0.80)
            status_factor = 1.0 if chunk.status == "current" else 0.60
            combined = (
                (0.56 * lexical_score + 0.44 * max(dense_score, 0.0))
                if query_vector is not None
                else lexical_score
            )
            combined *= authority * status_factor
            scored.append(SourceChunk(**{**asdict(chunk), "score": combined}))

        scored.sort(key=lambda item: item.score, reverse=True)

        # Exact product rows are essential when the question names a SKU. Avoid
        # allowing similar catalog rows to crowd policy chunks out of the top K.
        exact_product_ids = {f"products.csv:{sku}" for sku in extract_skus(query)}
        chosen: list[SourceChunk] = []
        catalog_count = 0
        for item in scored:
            is_catalog = item.doc_type in {"catalog", "serviceability"}
            if is_catalog and exact_product_ids and item.source_id not in exact_product_ids:
                continue
            if is_catalog and not exact_product_ids and catalog_count >= 2:
                continue
            chosen.append(item)
            catalog_count += int(is_catalog)
            if len(chosen) >= top_k:
                break

        # Precedence is a global retrieval fact and prevents lower-authority FAQ
        # or marketing text from silently overriding current policy.
        precedence = next((item for item in scored if item.source_id == "KB-00.1"), None)
        if precedence and all(item.source_id != "KB-00.1" for item in chosen):
            chosen[-1] = precedence

        for item in scored:
            if item.source_id in exact_product_ids and all(
                existing.source_id != item.source_id for existing in chosen
            ):
                chosen.insert(0, item)
        return chosen[:top_k]

    def prompt_context(self, sources: Iterable[SourceChunk]) -> str:
        return "\n\n".join(source.as_prompt_text() for source in sources)

