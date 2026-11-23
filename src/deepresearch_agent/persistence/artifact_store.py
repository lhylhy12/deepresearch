# 本地工件文件存储。报告、原始检索响应和轨迹等内容保存在文件中，数据库只保存引用及校验元数据。
"""Content-addressed, path-confined artifact writes."""

import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from deepresearch_agent.harness.errors import AppError, ErrorCode


@dataclass(frozen=True)
class StoredArtifact:
    # 工件的唯一标识，可据此定位保存的报告或原始工具响应。
    artifact_id: str
    # 相对于工件根目录的路径，可用于定位文件而不依赖部署绝对路径。
    relative_path: str
    # 工件内容的媒体类型，便于下载或预览。
    mime_type: str
    # 工件文件的字节大小。
    size_bytes: int
    # 原内容的 SHA-256，用于校验文件或待编辑片段是否变化。
    sha256: str


class ArtifactStore:
    # 将 root 解析为工件根目录，并确保它可供后续写入。
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # 将相对路径限定在工件根目录内，防止传入的路径跳出存储范围。
    def _resolve(self, relative_path: str | Path) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise AppError(ErrorCode.ARTIFACT_PATH_INVALID, "Artifact 路径超出允许根目录") from exc
        return candidate

    # 把字节写入指定工件，返回相对路径、媒体类型、大小和内容摘要等元数据。
    def write_bytes(self, relative_path: str | Path, content: bytes, *, mime_type: str = "application/octet-stream") -> StoredArtifact:
        target = self._resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(prefix=".artifact-", dir=str(target.parent))
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        except Exception:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
            raise
        digest = hashlib.sha256(content).hexdigest()
        return StoredArtifact(
            artifact_id=f"art_{uuid.uuid4().hex}",
            relative_path=target.relative_to(self.root).as_posix(),
            mime_type=mime_type,
            size_bytes=len(content),
            sha256=digest,
        )

    # 用 UTF-8 编码文本，再使用字节写入路径保存工件。
    def write_text(self, relative_path: str | Path, content: str, *, mime_type: str = "text/plain; charset=utf-8") -> StoredArtifact:
        return self.write_bytes(relative_path, content.encode("utf-8"), mime_type=mime_type)

    # 重算工件内容哈希并与预期 SHA-256 比较，用于校验文件完整性。
    def verify(self, relative_path: str | Path, expected_sha256: str) -> bool:
        target = self._resolve(relative_path)
        if not target.is_file():
            return False
        return hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256

    # 经过路径边界校验后读取工件原始字节。
    def read_bytes(self, relative_path: str | Path) -> bytes:
        return self._resolve(relative_path).read_bytes()
