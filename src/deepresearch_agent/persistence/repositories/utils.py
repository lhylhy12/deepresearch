import json
import uuid
from datetime import datetime, timezone
from typing import Any


# 将当前 UTC 时间格式化为持久层使用的 ISO 字符串。
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# 生成带业务类型前缀的唯一标识。
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# 把结构化数据统一序列化为可写入数据库文本列的 JSON。
def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)
