# Tavily 联网检索适配器：统一参数、重试、响应清理、磁盘缓存和稳定证据转换。
"""Tavily implementation with bounded retry, cache and stable Web evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tavily import TavilyClient

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalMetadata, RetrievalResult
from deepresearch_agent.config.settings import CACHE_DIR
from deepresearch_agent.harness.contracts import SourceMode
from deepresearch_agent.harness.errors import AppError, ErrorCode
from deepresearch_agent.harness.policies import SourcePolicy
from deepresearch_agent.persistence.artifact_store import ArtifactStore
from deepresearch_agent.retrieval.base import SearchDepth, SearchFilters, ToolCallContext
from deepresearch_agent.retrieval.web_utils import content_hash, domain_from_url, normalize_content, normalize_url


class TavilyProvider:
    mode = SourceMode.WEB
    provider_name = "tavily"
    _secret_pattern = re.compile(r"(?<![A-Za-z0-9_])(?:sk|tvly)-[A-Za-z0-9_-]{8,}|\bBearer\s+[A-Za-z0-9._-]{8,}|\bapi[_ -]?key\s*[:=]\s*\S+", re.I)

    # 初始化联网客户端和原始响应工件存储，并配置结果数、检索深度、超时和缓存有效期。
    def __init__(
        self,
        *,
        api_key: str,
        client: Optional[Any] = None,
        artifact_store: Optional[ArtifactStore] = None,
        cache_dir: str | Path | None = None,
        default_depth: SearchDepth = "advanced",
        default_max_results: int = 5,
        timeout_seconds: int = 45,
        cache_ttl_seconds: int = 86400,
        policy: Optional[SourcePolicy] = None,
    ) -> None:
        if not api_key and client is None:
            raise AppError(ErrorCode.TAVILY_API_KEY_MISSING, "联网搜索未配置 TAVILY_API_KEY")
        self._client = client or TavilyClient(api_key=api_key)
        self._artifact_store = artifact_store
        self._cache_dir = Path(cache_dir or (CACHE_DIR / "tavily")).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._default_depth = default_depth
        self._default_max_results = default_max_results
        self._timeout_seconds = timeout_seconds
        self._cache_ttl_seconds = cache_ttl_seconds
        self._policy = policy or SourcePolicy()

    # 清理联网请求参数；优先读有效缓存，未命中时调用搜索接口，再保存工件并返回标准化证据。
    async def search(
        self,
        query: str,
        *,
        top_k: int,
        search_depth: SearchDepth,
        filters: SearchFilters,
        call_context: ToolCallContext,
    ) -> list[RetrievalResult]:
        self._policy.assert_tool_allowed(call_context.source_mode, "tavily_search")
        args = self._policy.sanitize_web_arguments({
            "query": query,
            "top_k": top_k or self._default_max_results,
            "search_depth": search_depth or self._default_depth,
            "include_domains": list(filters.include_domains),
            "exclude_domains": list(filters.exclude_domains),
            "start_date": filters.start_date,
            "end_date": filters.end_date,
            "topic": filters.topic,
        })
        # 用过滤后的搜索参数查缓存；时效性查询采用更短 TTL，命中时不再次向搜索方发请求。
        cached = self._read_cache(args)
        if cached is None:
            raw = self._sanitize_external(await self._search_with_retry(args))
            self._write_cache(args, raw)
        else:
            raw = self._sanitize_external(cached)
        artifact = self._store_raw(call_context, raw)
        return self._map_results(raw, artifact=artifact, query=str(args["query"]))[: int(args["top_k"])]

    # 执行联网请求；对允许重试的失败等待后重试，避免把所有错误视为临时故障。
    async def _search_with_retry(self, args: dict[str, Any]) -> dict[str, Any]:
        call_args = {
            "query": args["query"],
            "search_depth": args["search_depth"],
            "max_results": args["top_k"],
            "include_answer": False,
            "include_raw_content": "markdown",
            "timeout": self._timeout_seconds,
        }
        for name in ("include_domains", "exclude_domains", "start_date", "end_date", "topic"):
            if args.get(name):
                call_args[name] = args[name]
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(self._client.search, **call_args)
                if not isinstance(response, dict):
                    raise TypeError("Tavily 返回值不是对象")
                return response
            except Exception as exc:  # Tavily SDK and requests expose several error families
                status = self._status_code(exc)
                name = type(exc).__name__.lower()
                if status in {401, 403} or any(token in name for token in ("invalidapikey", "forbidden", "missingapikey")):
                    raise AppError(ErrorCode.TAVILY_AUTH_FAILED, "Tavily 认证失败，请检查后端 API Key") from exc
                if status == 429 or "limit" in name or "ratelimit" in name:
                    if attempt == 2:
                        raise AppError(ErrorCode.TAVILY_RATE_LIMITED, "Tavily 请求频率受限", retryable=True) from exc
                    await asyncio.sleep(self._retry_delay(exc, attempt))
                    continue
                if status is not None and status < 500:
                    raise AppError(ErrorCode.RETRIEVAL_FAILED, "Tavily 请求失败", details={"status_code": status}) from exc
                is_timeout = "timeout" in name
                if attempt == 2:
                    code = ErrorCode.RETRIEVAL_TIMEOUT if is_timeout else ErrorCode.RETRIEVAL_FAILED
                    raise AppError(code, "Tavily 检索超时" if is_timeout else "Tavily 检索暂时不可用", retryable=True) from exc
                await asyncio.sleep((2 ** (attempt + 1)) + random.uniform(0, 0.25))
        raise AssertionError("unreachable")

    # 从异常或附带响应中获取 HTTP 状态码，供重试判断使用。
    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        value = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        return value or getattr(response, "status_code", None)

    # 优先使用服务端给出的重试等待时间；缺失或不可解析时采用指数退避。
    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        raw = headers.get("Retry-After") or headers.get("retry-after")
        try:
            return max(0.0, float(raw)) if raw is not None else float(2 ** (attempt + 1))
        except (TypeError, ValueError):
            return float(2 ** (attempt + 1))

    # 把网页结果变为证据，补充来源和工件引用；按规范化地址与内容去重，再按权威性和相关度排序。
    def _map_results(self, raw: dict[str, Any], *, artifact: Optional[dict[str, str]], query: str = "") -> list[RetrievalResult]:
        deduped: dict[tuple[str, str], RetrievalResult] = {}
        retrieved_at = datetime.now(timezone.utc)
        for item in raw.get("results", []) or []:
            try:
                url = normalize_url(str(item.get("url") or ""))
            except ValueError:
                continue
            body = str(item.get("raw_content") or item.get("content") or "")
            normalized = normalize_content(body)
            digest = content_hash(normalized)
            published = self._parse_datetime(item.get("published_date") or item.get("published_at"))
            score = min(1.0, max(0.0, float(item.get("score") or 0.5)))
            extra = {
                "untrusted_external_content": True,
                "content_truncated": len(normalized) > 12000,
            }
            domain = domain_from_url(url)
            query_terms = {
                token for token in re.findall(r"[a-z][a-z0-9-]{3,}", query.lower())
                if token not in {"basic", "advanced", "search", "depth", "official", "documentation", "document", "docs"}
            }
            brand_match = any(token in domain.lower().split(".") for token in query_terms)
            documentation_domain = domain.lower().startswith(("docs.", "help.", "developer."))
            public_authority = domain.lower().endswith((".gov", ".gov.cn", ".edu", ".edu.cn"))
            extra["authority_rank"] = 3 if brand_match and documentation_domain else 2 if documentation_domain or public_authority else 0
            if artifact:
                extra.update(artifact)
            metadata = RetrievalMetadata(
                source_id=url,
                source_type="webpage",
                title=item.get("title"),
                url=url,
                domain=domain,
                published_at=published,
                retrieved_at=retrieved_at,
                timestamp=published or retrieved_at,
                content_hash=digest,
                confidence=score,
                extra=extra,
            )
            result = RetrievalResult(
                granularity="DO",
                evidence=normalized[:12000],
                metadata=metadata,
                source="tavily_search",
                source_mode="web",
                score=score,
            )
            key = (url, digest)
            prior = deduped.get(key)
            if prior is None or result.score > prior.score:
                deduped[key] = result
        # 构造网页证据的排序键，先按来源等级，再按检索相关度。
        def authority(item: RetrievalResult) -> tuple[int, float]:
            return int((item.metadata.extra or {}).get("authority_rank") or 0), item.score

        return sorted(deduped.values(), key=authority, reverse=True)

    # 把当前工具调用的清理后响应存成工件，返回证据可追溯的工件 ID 和相对路径。
    def _store_raw(self, context: ToolCallContext, raw: dict[str, Any]) -> Optional[dict[str, str]]:
        if self._artifact_store is None:
            return None
        call_id = context.tool_call_id or hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:16]
        stored = self._artifact_store.write_text(
            f"{context.run_id}/tavily/{call_id}.json",
            json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str),
            mime_type="application/json",
        )
        return {"artifact_id": stored.artifact_id, "artifact_path": stored.relative_path}

    # 用排序后的搜索参数计算稳定哈希，使相同查询条件命中同一缓存。
    def _cache_key(self, args: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    # 读取未过期的字典缓存；新闻和时效性查询使用更短的有效期。
    def _read_cache(self, args: dict[str, Any]) -> Optional[dict[str, Any]]:
        path = self._cache_dir / f"{self._cache_key(args)}.json"
        if not path.is_file():
            return None
        ttl = 3600 if args.get("topic") == "news" or any(word in str(args["query"]).lower() for word in ("today", "latest", "最新", "今天")) else self._cache_ttl_seconds
        if time.time() - path.stat().st_mtime > ttl:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    # 将联网搜索的响应写入参数哈希对应的缓存文件。
    def _write_cache(self, args: dict[str, Any], raw: dict[str, Any]) -> None:
        path = self._cache_dir / f"{self._cache_key(args)}.json"
        temporary = path.with_suffix(f".{time.time_ns()}.tmp")
        temporary.write_text(json.dumps(raw, ensure_ascii=False, default=str), encoding="utf-8")
        temporary.replace(path)

    # 递归清理外部响应中的疑似密钥及敏感字段，再用于缓存或落库。
    @classmethod
    def _sanitize_external(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._secret_pattern.sub("[REDACTED]", value)
        if isinstance(value, list):
            return [cls._sanitize_external(item) for item in value]
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                if key.lower() == "url" and isinstance(item, str):
                    try:
                        sanitized[key] = normalize_url(item)
                    except ValueError:
                        sanitized[key] = ""
                else:
                    sanitized[key] = cls._sanitize_external(item)
            return sanitized
        return value

    # 将网页日期转换为带时区的时间；无法识别时返回 None。
    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None


__all__ = ["TavilyProvider"]
