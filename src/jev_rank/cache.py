from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_key(
    dataset: str,
    fingerprint: str,
    split: str,
    query_id: str,
    document_id: str,
    method: str,
    payload: Mapping[str, Any],
    requested_model: str,
) -> str:
    identity = {
        "dataset": dataset,
        "fingerprint": fingerprint,
        "split": split,
        "query_id": query_id,
        "document_id": document_id,
        "method": method,
        "payload": payload,
        "requested_model": requested_model,
    }
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


class JsonlCache:
    def __init__(self, path: Path):
        self.path = path
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def load(self) -> None:
        self._records.clear()
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A process killed during append may leave only the final record truncated.
                    if not any(rest.strip() for rest in handle):
                        break
                    raise ValueError(f"Malformed cache JSON on line {line_number}")
                if record.get("success") is True and record.get("cache_key"):
                    self._records[record["cache_key"]] = record

    def get(self, key: str) -> dict[str, Any] | None:
        return self._records.get(key)

    async def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(6):
                try:
                    with self.path.open("a", encoding="utf-8", newline="") as handle:
                        handle.write(line)
                        handle.flush()
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    await asyncio.sleep(0.05 * (attempt + 1))
            if record.get("success") is True:
                self._records[record["cache_key"]] = record
