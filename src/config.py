"""Configuration and validation for the capstone project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Allows offline validation before dependencies are installed.
    def load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    dataset_dir: Path
    openai_api_key: str
    openai_model: str
    embedding_model: str
    gnani_api_key: str
    reference_date: date = date(2026, 10, 20)

    @property
    def kb_dir(self) -> Path:
        return self.dataset_dir / "kb"

    @property
    def data_dir(self) -> Path:
        return self.dataset_dir / "data"

    @property
    def practice_dir(self) -> Path:
        return self.dataset_dir / "practice"

    @property
    def tools_file(self) -> Path:
        return self.dataset_dir / "tools" / "tools.py"

    def validate_dataset(self) -> None:
        required = [
            self.dataset_dir / "README.md",
            self.dataset_dir / "DATA_DICTIONARY.md",
            self.kb_dir,
            self.data_dir / "orders.csv",
            self.data_dir / "order_items.csv",
            self.data_dir / "products.csv",
            self.practice_dir / "questions.csv",
            self.practice_dir / "conversations.csv",
            self.tools_file,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            lines = "\n".join(f"  - {item}" for item in missing)
            raise FileNotFoundError(
                "The extracted course dataset was not found. Missing:\n"
                f"{lines}\nSet DATASET_DIR in .env to the extracted dataset folder."
            )


def get_settings(dataset_dir: str | Path | None = None) -> Settings:
    configured = dataset_dir or os.getenv("DATASET_DIR", "dataset")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return Settings(
        dataset_dir=path,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
        embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ).strip(),
        gnani_api_key=os.getenv("GNANI_API_KEY", "").strip(),
    )

