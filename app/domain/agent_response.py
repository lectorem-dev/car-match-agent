from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SessionUpdate(BaseModel):
    """Данные, которые агент извлек из сообщения пользователя.

    Важно:
    все поля обязательны для JSON-ответа LLM.
    Если данных нет — модель должна явно вернуть null, пустую строку или пустой список.
    """

    budget_min: Optional[int] = Field(
        ...,
        ge=0,
        description=(
            "Нижняя граница бюджета в долларах. "
            "Если пользователь не указал нижнюю границу бюджета, верни null."
        ),
    )

    budget_max: Optional[int] = Field(
        ...,
        ge=0,
        description=(
            "Верхняя граница бюджета в долларах. "
            "Например: 'до 15000 долларов' -> 15000. "
            "Если бюджет неизвестен, верни null."
        ),
    )

    purpose: Optional[str] = Field(
        ...,
        description=(
            "Цель покупки машины. "
            "Извлекай сценарий использования: первая машина, город, семья, работа, поездки, трасса. "
            "Например: 'первая машина для города' -> 'first car for city'. "
            "Если пользователь хочет подобрать машину, но цель слишком общая, верни 'buy suitable car'. "
            "Если цель вообще неясна, верни null."
        ),
    )

    experience_level: Optional[str] = Field(
        ...,
        description=(
            "Опыт водителя. "
            "Если пользователь ищет первую машину, верни 'beginner'. "
            "Если опыт неизвестен, верни null."
        ),
    )

    family_size: Optional[int] = Field(
        ...,
        ge=1,
        description=(
            "Размер семьи или количество людей, если пользователь явно указал. "
            "Если размер семьи неизвестен, верни null."
        ),
    )

    preferred_body_types: List[str] = Field(
        ...,
        description=(
            "Желаемые типы кузова: sedan, hatchback, liftback, wagon, suv, coupe, minivan, pickup. "
            "Если пользователь не указал тип кузова, верни пустой список []."
        ),
    )

    preferred_brands: List[str] = Field(
        ...,
        description=(
            "Желаемые бренды автомобилей. "
            "Если пользователь не указал бренды, верни пустой список []."
        ),
    )

    must_have: List[str] = Field(
        ...,
        description=(
            "Важные требования пользователя: automatic transmission, awd, electric, low mileage и т.д. "
            "Если требований нет, верни пустой список []."
        ),
    )

    must_not_have: List[str] = Field(
        ...,
        description=(
            "Только прямые запреты пользователя. "
            "Не добавляй сюда то, что пользователь просто не упомянул. "
            "Если запретов нет, верни пустой список []."
        ),
    )

    user_notes: str = Field(
        ...,
        description=(
            "Дополнительные нюансы из сообщения пользователя, если они не попали в отдельные поля. "
            "Если дополнительных нюансов нет, верни пустую строку ''."
        ),
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
        ...,
        description=(
            "Изменения, которые нужно применить к UserSession. "
            "Поле обязательно. Если новых данных нет, верни null для скалярных полей, "
            "[] для списков и '' для user_notes."

        ),
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