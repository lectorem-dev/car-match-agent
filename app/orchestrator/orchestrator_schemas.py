from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class RecommendedCar(BaseModel):
    """Одна машина в финальном ответе пайплайна."""

    car_id: UUID = Field(
        ...,
        description="UUID машины из каталога.",
    )

    reason: str = Field(
        ...,
        description="Почему машина подходит пользователю.",
    )

    risk_note: Optional[str] = Field(
        default=None,
        description="Возможный минус или ограничение машины.",
    )


class PipelineResponse(BaseModel):
    """Финальный ответ пайплайна пользователю."""

    user_message: str = Field(
        ...,
        description="Текст, который покажем пользователю.",
    )

    should_ask_clarifying_question: bool = Field(
        default=False,
        description="True, если агент должен задать уточняющий вопрос.",
    )

    clarifying_question: Optional[str] = Field(
        default=None,
        description="Текст уточняющего вопроса.",
    )

    recommended_cars: List[RecommendedCar] = Field(
        default_factory=list,
        min_length=0,
        max_length=5,
        description="Список рекомендованных машин.",
    )

    selected_car_id: Optional[UUID] = Field(
        default=None,
        description="UUID выбранной машины, если пользователь выбрал конкретный автомобиль.",
    )

    ready_for_reservation: bool = Field(
        default=False,
        description="True, если создана mock-заявка на бронь.",
    )
