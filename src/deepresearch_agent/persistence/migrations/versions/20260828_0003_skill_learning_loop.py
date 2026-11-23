"""durable skill learning loop

Revision ID: 20260828_0003
Revises: 20260823_0002
"""

from alembic import op

from deepresearch_agent.persistence.models import LearningReviewJobModel, SkillDeploymentModel, SkillReadMarkModel

revision = "20260828_0003"
down_revision = "20260823_0002"
branch_labels = None
depends_on = None


# 创建技能学习任务、技能读取记录和分阶段部署所需结构。
def upgrade() -> None:
    bind = op.get_bind()
    LearningReviewJobModel.__table__.create(bind, checkfirst=True)
    SkillReadMarkModel.__table__.create(bind, checkfirst=True)
    SkillDeploymentModel.__table__.create(bind, checkfirst=True)


# 撤销后台技能学习及部署相关表结构。
def downgrade() -> None:
    bind = op.get_bind()
    SkillDeploymentModel.__table__.drop(bind, checkfirst=True)
    SkillReadMarkModel.__table__.drop(bind, checkfirst=True)
    LearningReviewJobModel.__table__.drop(bind, checkfirst=True)
