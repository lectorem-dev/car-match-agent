from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PlannedCar(BaseModel):
    """Одна машина, выбранная PlannerAgent."""

    car_id: UUID
    reason: str = Field(max_length=80)  # Короткая машинная причина выбора.
    risk_note: Optional[str] = None  # Возможный минус машины.


class PlannerResult(BaseModel):
    """Результат подбора машин."""

    recommendations: List[PlannedCar] = Field(
        default_factory=list,
        min_length=1,
        max_length=3,
    )

    user_message: str = ""  # Техническое поле для совместимости, не используется в финальном ответе.
