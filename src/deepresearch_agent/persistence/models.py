# SQLite 的 ORM 模型。Run 是执行记录的中心，任务、工具调用、证据、检查点和验证项都可由其串回同一次研究。
"""SQLAlchemy models for the local MVP's durable fact store."""

from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# 一条会话记录，容纳多次运行，并保存摘要、归档状态及冻结的记忆快照。
class SessionModel(Base):
    __tablename__ = "sessions"
    # 对话会话的标识，用于关联消息、运行与记忆快照。
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 用于展示或组织内容的标题。
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    # 序列化的会话摘要，供长会话压缩后还原历史信息。
    summary_json: Mapped[str | None] = mapped_column(Text)
    # 会话冻结时保存的长期记忆 JSON，后续轮次沿用同一快照。
    memory_snapshot_json: Mapped[str | None] = mapped_column(Text)
    # 会话所用记忆快照的版本。
    memory_snapshot_version: Mapped[int | None] = mapped_column(Integer)
    # 此会话冻结长期记忆快照的时间。
    memory_snapshot_created_at: Mapped[str | None] = mapped_column(String(40))
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 最近一次更新记录的时间。
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 记录被归档的时间；未归档时可为空。
    archived_at: Mapped[str | None] = mapped_column(String(40))
    # 软删除时间；原记录保留用于历史追溯。
    deleted_at: Mapped[str | None] = mapped_column(String(40))
    __table_args__ = (Index("ix_sessions_status_updated", "status", "updated_at"),)


# 一条对话消息；同一会话中的客户端消息标识用于去重，每次运行最多交付一条助手回复。
class MessageModel(Base):
    __tablename__ = "messages"
    # 单条对话消息的标识，用于引用历史上下文。
    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 对话会话的标识，用于关联消息、运行与记忆快照。
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), nullable=False)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str | None] = mapped_column(String(64))
    # 客户端为一次提交生成的标识，与 session_id 组合进行幂等去重。
    client_message_id: Mapped[str | None] = mapped_column(String(128))
    # 消息发送方角色，如 user、assistant 或 system。
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    # 需要保存、检索或传入模型的文本内容。
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 与正文或证据一起存放的扩展 JSON，如来源或过程信息。
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (
        CheckConstraint("role IN ('user','assistant','system')", name="ck_messages_role"),
        UniqueConstraint("session_id", "client_message_id", name="uq_messages_client_id"),
        Index("ix_messages_session_created", "session_id", "created_at"),
        Index("uq_messages_assistant_run", "run_id", unique=True, sqlite_where=text("role = 'assistant' AND run_id IS NOT NULL")),
    )


# 一次研究运行的主记录，保存输入关联、来源、生命周期、预算和执行租约。
class RunModel(Base):
    __tablename__ = "runs"
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 对话会话的标识，用于关联消息、运行与记忆快照。
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), nullable=False)
    # 创建本轮运行的用户消息 ID，用于还原本轮输入及建立关联。
    trigger_message_id: Mapped[str] = mapped_column(ForeignKey("messages.message_id"), nullable=False, unique=True)
    # 本轮锁定的信息源模式，graphrag 对应私域，web 对应联网。
    source_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    # 指定研究执行流程，例如多轮深度研究或规划、执行、报告。
    workflow_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    # 运行当前业务阶段，用于进度展示和恢复定位。
    current_stage: Mapped[str | None] = mapped_column(String(32))
    # 运行配置快照的 JSON 存储形式。
    config_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    # 模型配置及选择结果的 JSON 存储形式。
    model_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # 运行创建时的预算上限配置。
    budget_json: Mapped[str] = mapped_column(Text, nullable=False)
    # 最近一次落库的预算和累计用量。
    usage_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # 协作取消请求标记；由运行时在检查边界处理。
    cancellation_requested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 持有该运行租约的执行者标识，空值表示尚无持有者。
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    # 租约过期时间，用于判断崩溃或失联后是否可恢复。
    lease_expires_at: Mapped[str | None] = mapped_column(String(40))
    # 供程序识别的错误分类码。
    error_code: Mapped[str | None] = mapped_column(String(64))
    # 记录供用户查看或复盘的失败说明。
    error_message: Mapped[str | None] = mapped_column(Text)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 开始执行或进入部署阶段的时间。
    started_at: Mapped[str | None] = mapped_column(String(40))
    # 运行或任务处理结束的时间；处理中可为空。
    completed_at: Mapped[str | None] = mapped_column(String(40))
    # 最近一次更新记录的时间。
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (
        CheckConstraint("source_mode IN ('graphrag','web')", name="ck_runs_source_mode"),
        CheckConstraint("workflow_mode IN ('deep_research','plan_execute_report')", name="ck_runs_workflow_mode"),
        Index("ix_runs_session_created", "session_id", "created_at"),
        Index("ix_runs_status_lease", "status", "lease_expires_at"),
    )


# 可重放的运行事件；持久化事件序号用于客户端断线后补齐进度。
class RunEventModel(Base):
    __tablename__ = "run_events"
    # 自增事件序号，作为进度事件断线重放的游标。
    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    # 标识事件类别，例如阶段切换、证据入账或任务完成。
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    # 记录该事件、快照或部署所处的阶段。
    stage: Mapped[str | None] = mapped_column(String(32))
    # 按 JSON 保存的扩展事件或业务载荷。
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    # 序列化结构的版本，用于兼容读取和迁移。
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (Index("ix_events_run_event", "run_id", "event_id"),)


# 一个运行阶段的版本化恢复快照，含数据格式版本和内容校验哈希。
class CheckpointModel(Base):
    __tablename__ = "checkpoints"
    # 一条持久化检查点的唯一标识。
    checkpoint_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    # 该数据或技能的版本号，具体递增规则由其类型决定。
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    # 记录该事件、快照或部署所处的阶段。
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    # 序列化的运行状态和工作流数据，用于恢复阶段。
    state_json: Mapped[str] = mapped_column(Text, nullable=False)
    # 状态 JSON 的指纹，用于检查快照完整性。
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # 序列化结构的版本，用于兼容读取和迁移。
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (UniqueConstraint("run_id", "version", name="uq_checkpoints_run_version"), Index("ix_checkpoints_run_version", "run_id", "version"))


# 一份版本化研究计划；同一运行可因重新规划保存多份计划。
class PlanModel(Base):
    __tablename__ = "plans"
    # 一份计划的标识，用于连接计划与其任务。
    plan_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    # 该数据或技能的版本号，具体递增规则由其类型决定。
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    # 保存计划的完整结构和执行要求。
    plan_json: Mapped[str] = mapped_column(Text, nullable=False)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 最近一次更新记录的时间。
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (UniqueConstraint("run_id", "version", name="uq_plans_run_version"),)


# 计划中的一项任务，记录任务类型、依赖描述、运行归属和完成状态。
class TaskModel(Base):
    __tablename__ = "tasks"
    # 任务标识，连接任务记录、工具调用与产出的证据。
    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    # 一份计划的标识，用于连接计划与其任务。
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.plan_id"), nullable=False)
    # 用于选择 Worker 或工具的任务类型。
    task_type: Mapped[str] = mapped_column(String(48), nullable=False)
    # 本轮锁定的信息源模式，graphrag 对应私域，web 对应联网。
    source_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    # 任务的依赖、输入及执行结果等结构化描述。
    task_json: Mapped[str] = mapped_column(Text, nullable=False)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 最近一次更新记录的时间。
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (Index("ix_tasks_run_status", "run_id", "status"), CheckConstraint("source_mode IN ('graphrag','web')", name="ck_tasks_source_mode"))


# 一次工具调用轨迹，记录实际参数、返回值或错误，供重放、诊断和去重。
class ToolCallModel(Base):
    __tablename__ = "tool_calls"
    # 工具调用的稳定标识，用于幂等查询及证据溯源。
    tool_call_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    # 任务标识，连接任务记录、工具调用与产出的证据。
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.task_id"))
    # 实际调用的工具名称，供审计和来源策略检查。
    tool_name: Mapped[str] = mapped_column(String(80), nullable=False)
    # 本轮锁定的信息源模式，graphrag 对应私域，web 对应联网。
    source_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    # 执行工具时使用的参数快照。
    args_json: Mapped[str] = mapped_column(Text, nullable=False)
    # 工具调用完成后保存的结构化结果。
    result_json: Mapped[str | None] = mapped_column(Text)
    # 供程序识别的错误分类码。
    error_code: Mapped[str | None] = mapped_column(String(64))
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 运行或任务处理结束的时间；处理中可为空。
    completed_at: Mapped[str | None] = mapped_column(String(40))
    __table_args__ = (Index("ix_tool_calls_run_task", "run_id", "task_id"), CheckConstraint("source_mode IN ('graphrag','web')", name="ck_tool_calls_source_mode"))


# 一条已登记的证据，关联工具、任务与原始内容工件；报告使用 evidence_id 引用。
class EvidenceModel(Base):
    __tablename__ = "evidence"
    # 账本中的稳定证据标识，报告通过该 ID 引用证据。
    evidence_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    # 任务标识，连接任务记录、工具调用与产出的证据。
    task_id: Mapped[str | None] = mapped_column(String(64))
    # 工具调用的稳定标识，用于幂等查询及证据溯源。
    tool_call_id: Mapped[str | None] = mapped_column(String(64))
    # 本轮锁定的信息源模式，graphrag 对应私域，web 对应联网。
    source_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    # 实际返回证据的检索提供方标识。
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    # 原始资料标识，例如规范化网页地址或图谱文档/文本块 ID；与本轮引用 ID 分开。
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    # 用于展示或组织内容的标题。
    title: Mapped[str | None] = mapped_column(Text)
    # 对原始内容或运行片段的简短概括。
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    # 与正文或证据一起存放的扩展 JSON，如来源或过程信息。
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # 内容的稳定指纹，用于重复检测、缓存复用或完整性校验。
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # 工件的唯一标识，可据此定位保存的报告或原始工具响应。
    artifact_id: Mapped[str | None] = mapped_column(String(64))
    # 用于排序或判定的相关度分数，含义由对应对象定义。
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    # 证据被标为无效的时间。
    invalidated_at: Mapped[str | None] = mapped_column(String(40))
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (Index("ix_evidence_run_source", "run_id", "source_mode"), Index("ix_evidence_content_hash", "content_hash"), CheckConstraint("source_mode IN ('graphrag','web')", name="ck_evidence_source_mode"))


# 本轮完成契约中的一项检查，保留是否强制、阈值、观察值和通过结果。
class ContractCheckModel(Base):
    __tablename__ = "contract_checks"
    # 完成契约单项检查结果的唯一标识。
    check_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    # 该记录的业务分类，例如记忆类别或契约检查类型。
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    # 该项检查是否必须通过才允许完成运行。
    required: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # 验证门槛的序列化值，便于复查判定依据。
    threshold_json: Mapped[str | None] = mapped_column(Text)
    # 产生该检查结果的验证器名称。
    verifier: Mapped[str] = mapped_column(String(80), nullable=False)
    # 运行验证逻辑的版本，用来复盘校验标准变化。
    verifier_version: Mapped[str] = mapped_column(String(32), nullable=False)
    # 是否通过相应检查或门禁；部分持久记录允许为空表示未判定。
    passed: Mapped[int | None] = mapped_column(Integer)
    # 验证器保存的观察值、解释和证据引用。
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (Index("ix_contract_checks_run", "run_id"),)


# 文件工件的数据库索引；文件正文由 ArtifactStore 独立存储。
class ArtifactModel(Base):
    __tablename__ = "artifacts"
    # 工件的唯一标识，可据此定位保存的报告或原始工具响应。
    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.run_id"))
    # 相对于工件根目录的路径，可用于定位文件而不依赖部署绝对路径。
    relative_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # 工件内容的媒体类型，便于下载或预览。
    mime_type: Mapped[str] = mapped_column(String(120), nullable=False)
    # 工件文件的字节大小。
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    # 原内容的 SHA-256，用于校验文件或待编辑片段是否变化。
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


# 长期记忆记录；用目标和作用域区分适用范围，并保留来源、有效期及替代关系。
class MemoryModel(Base):
    __tablename__ = "memories"
    # 长期记忆的唯一标识。
    memory_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 对话会话的标识，用于关联消息、运行与记忆快照。
    session_id: Mapped[str | None] = mapped_column(ForeignKey("sessions.session_id"))
    # 长期记忆的归属目标，例如用户或项目。
    target: Mapped[str | None] = mapped_column(String(32))
    # 记忆适用的作用域，影响跨会话检索和冲突检查。
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    # 该记录的业务分类，例如记忆类别或契约检查类型。
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    # 需要保存、检索或传入模型的文本内容。
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 长期记忆来源引用的 JSON 表示。
    provenance_json: Mapped[str] = mapped_column(Text, nullable=False)
    # 该候选、匹配或判断的置信度，用于筛选和展示。
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    # 该记忆开始有效的时间。
    valid_from: Mapped[str] = mapped_column(String(40), nullable=False)
    # 记忆失效时间；空值表示不设过期时间。
    expires_at: Mapped[str | None] = mapped_column(String(40))
    # 被这条新记忆替代的旧记录标识。
    supersedes: Mapped[str | None] = mapped_column(ForeignKey("memories.memory_id"))
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    # 创建记录的用户、系统或后台服务。
    created_by: Mapped[str] = mapped_column(String(24), nullable=False)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 最近一次更新记录的时间。
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 软删除时间；原记录保留用于历史追溯。
    deleted_at: Mapped[str | None] = mapped_column(String(40))
    # 记录被归档的时间；未归档时可为空。
    archived_at: Mapped[str | None] = mapped_column(String(40))
    __table_args__ = (Index("ix_memories_status_scope_expiry", "status", "scope", "expires_at"),)


# 某个技能的特定版本，保存内容路径、哈希、状态和经验来源。
class SkillVersionModel(Base):
    __tablename__ = "skill_versions"
    # 数据库中的技能版本标识，区分同名技能的各个版本。
    skill_version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 对象的业务名称，通常用作展示或检索键。
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    # 该数据或技能的版本号，具体递增规则由其类型决定。
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    # 技能内容在注册表目录中的相对路径。
    content_path: Mapped[str] = mapped_column(Text, nullable=False)
    # 内容的稳定指纹，用于重复检测、缓存复用或完整性校验。
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # 支持该技能的来源运行 ID 列表的 JSON 形式。
    source_run_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (UniqueConstraint("name", "version", name="uq_skill_name_version"), Index("ix_skill_name_status", "name", "status"))


# 尚需评测或发布决策的技能候选，关联产生提案的研究运行。
class SkillCandidateModel(Base):
    __tablename__ = "skill_candidates"
    # 技能候选提案标识，可关联评审、评测和发布结果。
    candidate_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    # 对象的业务名称，通常用作展示或检索键。
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    # 候选提案拟登记的版本号。
    proposed_version: Mapped[str] = mapped_column(String(32), nullable=False)
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    # 按 JSON 保存的扩展事件或业务载荷。
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


# 一次针对技能候选的评测，保留结果、用量和质量指标。
class EvalRunModel(Base):
    __tablename__ = "eval_runs"
    # 一次候选技能评测的唯一标识。
    eval_run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 技能候选提案标识，可关联评审、评测和发布结果。
    candidate_id: Mapped[str] = mapped_column(ForeignKey("skill_candidates.candidate_id"), nullable=False)
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    # 评测指标或灰度部署观测指标的 JSON 形式。
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 运行或任务处理结束的时间；处理中可为空。
    completed_at: Mapped[str | None] = mapped_column(String(40))


# 由完成运行触发的后台评审任务，保存状态、重试和各阶段产物。
class LearningReviewJobModel(Base):
    __tablename__ = "learning_review_jobs"
    # 后台技能学习评审的标识，关联评审进度和产物。
    review_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 一次研究运行的标识，用来关联阶段、结果和运行记录。
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    # 触发该学习任务的运行终止事件 ID，用于追溯和防重复创建。
    terminal_event_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 建立评审任务时所用的学习策略版本。
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1")
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    # 评审已经完成的阶段及恢复所需状态。
    checkpoint_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # 评审材料包的持久化 JSON。
    review_pack_json: Mapped[str | None] = mapped_column(Text)
    # 提案模型生成的候选方案。
    proposal_json: Mapped[str | None] = mapped_column(Text)
    # 批评模型给出的审查结论。
    critic_json: Mapped[str | None] = mapped_column(Text)
    # 提案规则校验的结构化结果。
    validation_json: Mapped[str | None] = mapped_column(Text)
    # 技能候选提案标识，可关联评审、评测和发布结果。
    candidate_id: Mapped[str | None] = mapped_column(ForeignKey("skill_candidates.candidate_id"))
    # 累计发生的重试次数。
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 评审期间已生成的修订次数。
    revision_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 记录供用户查看或复盘的失败说明。
    error_message: Mapped[str | None] = mapped_column(Text)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 最近一次更新记录的时间。
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 运行或任务处理结束的时间；处理中可为空。
    completed_at: Mapped[str | None] = mapped_column(String(40))
    __table_args__ = (
        UniqueConstraint("run_id", "terminal_event_id", "policy_version", name="uq_learning_review_idempotency"),
        Index("ix_learning_review_status", "status", "updated_at"),
    )


# 记录某个评审读取过的技能版本与哈希，用于核对修订提案的基准。
class SkillReadMarkModel(Base):
    __tablename__ = "skill_read_marks"
    # 某次评审读取技能版本的记录标识。
    read_mark_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 后台技能学习评审的标识，关联评审进度和产物。
    review_id: Mapped[str] = mapped_column(ForeignKey("learning_review_jobs.review_id"), nullable=False)
    # 数据库中的技能版本标识，区分同名技能的各个版本。
    skill_version_id: Mapped[str] = mapped_column(ForeignKey("skill_versions.skill_version_id"), nullable=False)
    # 内容的稳定指纹，用于重复检测、缓存复用或完整性校验。
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (UniqueConstraint("review_id", "skill_version_id", name="uq_skill_read_mark"),)


# 记录技能版本的一次部署阶段及流量、成败和安全观测。
class SkillDeploymentModel(Base):
    __tablename__ = "skill_deployments"
    # 一条技能部署阶段记录的标识。
    deployment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 数据库中的技能版本标识，区分同名技能的各个版本。
    skill_version_id: Mapped[str] = mapped_column(ForeignKey("skill_versions.skill_version_id"), nullable=False)
    # 记录该事件、快照或部署所处的阶段。
    stage: Mapped[str] = mapped_column(String(24), nullable=False)
    # 分给该技能版本的运行流量比例，取百分比数值。
    allocation_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 记录当前生命周期状态；允许值由对应模型或状态机限定。
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="active")
    # 评测指标或灰度部署观测指标的 JSON 形式。
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # 开始执行或进入部署阶段的时间。
    started_at: Mapped[str] = mapped_column(String(40), nullable=False)
    # 该部署阶段停止生效的时间。
    stopped_at: Mapped[str | None] = mapped_column(String(40))


# 对记忆、技能等对象的持久操作审计记录。
class AuditEventModel(Base):
    __tablename__ = "audit_events"
    # 一条审计记录的唯一标识。
    audit_event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 这一步实际采取的动作，或恢复决策建议采取的动作。
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    # 审计操作所针对的对象类型。
    entity_type: Mapped[str] = mapped_column(String(48), nullable=False)
    # 审计操作所针对的对象标识。
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # 按 JSON 保存的扩展事件或业务载荷。
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # 记录创建时的时间戳。
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (Index("ix_audit_entity", "entity_type", "entity_id"),)
