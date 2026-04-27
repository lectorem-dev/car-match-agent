from typing import List
from uuid import UUID

from pydantic import BaseModel, Field


class CriticCarReview(BaseModel):
    """Результат проверки одной рекомендованной машины."""

    car_id: UUID
    approved: bool
    issues: List[str] = Field(default_factory=list)


class CriticResult(BaseModel):
    """Результат проверки рекомендаций."""

    approved: bool = False
    car_reviews: List[CriticCarReview] = Field(default_factory=list)
    approved_car_ids: List[UUID] = Field(default_factory=list)
    rejected_car_ids: List[UUID] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)  # Агрегированные проблемы для retry и логов.
    user_message: str = ""  # Финальное сообщение пользователю только по одобренным машинам.
