from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PlannedCarRecommendation(BaseModel):
    """Одна машина, выбранная PlannerAgent."""

    car_id: UUID
    reason: str  # Почему машина подходит пользователю.
    risk_note: Optional[str] = None  # Возможный минус машины.


class PlannerResult(BaseModel):
    """Результат подбора машин."""

    recommendations: List[PlannedCarRecommendation] = Field(
        default_factory=list,
        min_length=1,
        max_length=5,
    )

    user_message: str  # Черновик ответа пользователю.