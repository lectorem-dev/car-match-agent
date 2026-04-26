from typing import List

from pydantic import BaseModel, Field


class CriticResult(BaseModel):
    """Результат проверки рекомендаций."""

    approved: bool
    issues: List[str] = Field(default_factory=list)  # Найденные проблемы.
