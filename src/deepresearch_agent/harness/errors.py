"""Typed public errors; messages are safe to expose and never contain secrets."""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    CONFIGURATION_INVALID = "CONFIGURATION_INVALID"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    INVALID_SOURCE_MODE = "INVALID_SOURCE_MODE"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    LEASE_UNAVAILABLE = "LEASE_UNAVAILABLE"
    ARTIFACT_PATH_INVALID = "ARTIFACT_PATH_INVALID"
    ARTIFACT_INTEGRITY_FAILED = "ARTIFACT_INTEGRITY_FAILED"
    SOURCE_POLICY_VIOLATION = "SOURCE_POLICY_VIOLATION"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    TAVILY_API_KEY_MISSING = "TAVILY_API_KEY_MISSING"
    TAVILY_AUTH_FAILED = "TAVILY_AUTH_FAILED"
    TAVILY_RATE_LIMITED = "TAVILY_RATE_LIMITED"
    RETRIEVAL_TIMEOUT = "RETRIEVAL_TIMEOUT"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"


class AppError(RuntimeError):
    # 构造可跨 API 和运行时传递的错误；code 用于程序判断，message 用于展示，details 补充诊断信息。
    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}

    # 将错误信息转换成可序列化字典，保留是否允许重试。
    def as_dict(self) -> Dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "retryable": self.retryable, "details": self.details}


class RunCancelled(RuntimeError):
    """Internal cooperative-cancellation signal; never exposed as a failure."""
