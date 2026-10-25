"""Controlled access to the dataset-provided mock tools."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


class ToolLayer:
    def __init__(self, tools_file: Path):
        self.tools_file = tools_file
        self.module = self._load(tools_file)

    @staticmethod
    def _load(path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location("zentromart_dataset_tools", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load mock tools from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def lookup_order(self, order_id: str, verification_value: str | None = None) -> dict:
        if verification_value is None:
            return self.module.lookup_order(order_id)
        return self.module.lookup_order(order_id, verification_value=verification_value)

    def create_ticket(
        self,
        summary: str,
        order_id: str | None = None,
        reason: str | None = None,
        intent: str | None = None,
    ) -> dict:
        return self.module.create_ticket(
            summary=summary,
            order_id=order_id,
            reason=reason,
            intent=intent,
        )

