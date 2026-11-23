"""Compatibility import; compaction belongs to context engineering."""

from deepresearch_agent.context.compactor import ContextCompactor


class SessionSummarizer(ContextCompactor):
    """Legacy constructor mapped onto token-pressure compaction."""

    # 绑定会话和消息仓储；threshold_messages 指定触发会话摘要所需的消息数。
    def __init__(self, sessions, messages, *, threshold_messages: int = 20):
        super().__init__(
            sessions, messages,
            threshold_tokens=max(100, threshold_messages * 10),
            protect_recent_messages=max(2, threshold_messages // 2),
        )

__all__ = ["ContextCompactor", "SessionSummarizer"]
