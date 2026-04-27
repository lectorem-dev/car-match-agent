from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.user_session import UserSession


# Тип шага в тестовом сценарии.
class ScenarioStepType(str, Enum):
    USER_MESSAGE = "user_message"
    EXPECT_CLARIFYING_QUESTION = "expect_clarifying_question"
    EXPECT_RECOMMENDATION = "expect_recommendation"
    USER_SELECT_CAR = "user_select_car"
    USER_REQUEST_RESERVATION = "user_request_reservation"
    EXPECT_RESERVATION_CREATED = "expect_reservation_created"


class ScenarioStep(BaseModel):
    """Один шаг тестового сценария."""

    step_type: ScenarioStepType

    user_message: Optional[str] = None  # Сообщение пользователя.
    selected_car_id: Optional[UUID] = None  # Машина, которую выбирает пользователь.

    acceptable_car_ids: List[UUID] = Field(default_factory=list)  # Допустимые машины для рекомендации.


class TestScenario(BaseModel):
    """Полный тестовый сценарий работы агента."""

    id: str
    name: str

    allow_clarifying_question: bool = False  # Можно ли агенту задавать уточняющий вопрос.

    initial_session: Optional[UserSession] = None  # Состояние сессии до старта сценария.
    steps: List[ScenarioStep]
