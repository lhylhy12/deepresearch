"""Redacted trajectory export for legacy Plan–Execute–Report runs."""

import json
import re
from typing import Any

from .artifact_store import ArtifactStore, StoredArtifact

_SECRET_KEY = re.compile(r"(?i)(api[_-]?key|authorization|password|secret|token)")
_SECRET_VALUE = re.compile(r"(?i)(sk-[a-z0-9_-]{8,}|bearer\s+[a-z0-9._-]{8,})")


# 递归清理轨迹对象中的敏感字段，避免导出文件泄露配置或凭据。
def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    return value


class TrajectoryExporter:
    # 绑定工件存储，用于保存脱敏后的轨迹文件。
    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    # 将会话轨迹脱敏并导出到工件存储，返回工件描述。
    def export(self, *, session_id: str, payload: dict[str, Any]) -> StoredArtifact:
        safe_payload = redact(payload)
        serialized = json.dumps(safe_payload, ensure_ascii=False, indent=2, default=str)
        return self.artifact_store.write_text(
            f"legacy/{session_id}/trajectory.json",
            serialized,
            mime_type="application/json; charset=utf-8",
        )
