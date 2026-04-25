from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SessionUpdate(BaseModel):
    """Данные, которые агент извлек из сообщения пользователя."""

    budget_min: Optional[int] = Field(
        default=None,
        description="Нижняя граница бюджета в долларах, если пользователь ее указал.",
    )

    budget_max: Optional[int] = Field(
        default=None,
        description="Верхняя граница бюджета в долларах. Например: 'до 15000 долларов' -> 15000.",
    )

    purpose: Optional[str] = Field(
        default=None,
        description=(
            "Цель покупки машины. Заполняй, если пользователь указал сценарий использования: "
            "первая машина, для города, для семьи, для работы, для поездок, для трассы. "
            "Например: 'первая машина для города' -> 'first car for city'."
        ),
    )

    experience_level: Optional[str] = Field(
        default=None,
        description=(
            "Опыт водителя. Если пользователь ищет первую машину, заполни 'beginner'."
        ),
    )

    family_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Размер семьи, если пользователь явно указал количество людей.",
    )

    preferred_body_types: List[str] = Field(
        default_factory=list,
        description="Желаемые типы кузова: sedan, hatchback, liftback, wagon, suv, coupe, minivan, pickup.",
    )

    preferred_brands: List[str] = Field(
        default_factory=list,
        description="Желаемые бренды автомобилей, если пользователь явно их указал.",
    )

    must_have: List[str] = Field(
        default_factory=list,
        description=(
            "Важные требования пользователя. Например: automatic transmission, awd, electric, low mileage."
        ),
    )

    must_not_have: List[str] = Field(
        default_factory=list,
        description=(
            "Только прямые запреты пользователя. Не добавляй сюда то, что пользователь просто не упомянул."
        ),
    )

    user_notes: str = Field(
        default="",
        description="Дополнительные нюансы из сообщения пользователя, если они не попали в отдельные поля.",
    )


class CarRecommendation(BaseModel):
    """Одна машина в списке рекомендаций."""

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


class AgentResponse(BaseModel):
    """Структурированный ответ агента."""

    user_message: str = Field(
        ...,
        description="Текст, который покажем пользователю.",
    )

    session_update: Optional[SessionUpdate] = Field(
        default=None,
        description="Изменения, которые нужно применить к UserSession.",
    )

    should_ask_clarifying_question: bool = Field(
        default=False,
        description="True, если агент должен задать уточняющий вопрос.",
    )

    clarifying_question: Optional[str] = Field(
        default=None,
        description="Текст уточняющего вопроса.",
    )

    recommended_cars: List[CarRecommendation] = Field(
        default_factory=list,
        min_length=0,
        max_length=5,
        description="Список рекомендованных машин. На этапе рекомендации должно быть от 1 до 5 машин.",
    )

    selected_car_id: Optional[UUID] = Field(
        default=None,
        description="UUID выбранной машины, если пользователь выбрал конкретный автомобиль.",
    )

    ready_for_reservation: bool = Field(
        default=False,
        description="True, если можно переходить к созданию mock-заявки.",
    )