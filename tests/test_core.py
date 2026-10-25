from __future__ import annotations

import csv
import os
import unittest
from pathlib import Path

from src.config import get_settings
from src.knowledge import KnowledgeBase
from src.tools_layer import ToolLayer
from src.utils import (
    add_business_days,
    detect_reply_language,
    extract_order_id,
    extract_verification_value,
)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = os.getenv("DATASET_DIR")
        if not dataset:
            raise unittest.SkipTest("Set DATASET_DIR to run dataset tests")
        cls.settings = get_settings(dataset)
        cls.settings.validate_dataset()

    def test_number_extraction(self):
        self.assertEqual(extract_order_id("order 8978 2837"), "89782837")
        self.assertEqual(extract_order_id("eight nine seven eight two eight three seven"), "89782837")
        self.assertEqual(extract_verification_value("last four are 4533"), "4533")

    def test_language_detection(self):
        self.assertEqual(detect_reply_language("मेरा ऑर्डर कहाँ है?"), "hi")
        self.assertEqual(detect_reply_language("mera order late hai kya"), "hi-en")
        self.assertEqual(detect_reply_language("Where is my order?"), "en")

    def test_business_days_skip_sunday_and_holiday(self):
        from datetime import date

        result = add_business_days(
            date(2026, 10, 17), 5, [date(2026, 10, 19)]
        )
        self.assertEqual(result.isoformat(), "2026-10-24")

    def test_knowledge_base_preserves_sections(self):
        kb = KnowledgeBase(self.settings, use_embeddings=False)
        source_ids = {chunk.source_id for chunk in kb.chunks}
        self.assertIn("RET-2", source_ids)
        self.assertIn("REF-2", source_ids)
        self.assertIn("IDV-2", source_ids)
        self.assertGreaterEqual(len(source_ids), 100)

    def test_refund_retrieval(self):
        kb = KnowledgeBase(self.settings, use_embeddings=False)
        ids = [item.source_id for item in kb.retrieve("How long is a UPI refund?", top_k=8)]
        self.assertIn("REF-2", ids)

    def test_unverified_lookup_does_not_disclose_order(self):
        tools = ToolLayer(self.settings.tools_file)
        with (self.settings.data_dir / "orders.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            order_id = next(csv.DictReader(handle))["order_id"]
        result = tools.lookup_order(order_id)
        self.assertTrue(result["verification_required"])
        forbidden = {"status", "items", "total_inr", "delivery_pincode", "customer_id"}
        self.assertFalse(forbidden & result.keys())


if __name__ == "__main__":
    unittest.main()

