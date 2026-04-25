from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Состояние диалога.
class DialogStatus(str, Enum):
    # Первичный опрос: агент собирает базовые требования — бюджет, цель покупки, ограничения.
    INITIAL_SURVEY = "initial_survey"

    # Опциональный шаг: агент задает уточняющий вопрос.
    CLARIFYING_QUESTION = "clarifying_question"

    # Минимум данных собран, агент уже может предложить машины.
    READY_TO_RECOMMEND = "ready_to_recommend"

    # Пользователь выбрал конкретную машину из предложенных вариантов.
    CAR_SELECTED = "car_selected"

    # Пользователь подтвердил интерес к выбранной машине, можно создавать mock-заявку.
    READY_FOR_RESERVATION = "ready_for_reservation"

    # Mock-заявка создана.
    RESERVATION_CREATED = "reservation_created"


class UserSession(BaseModel):
    """Память пользователя в рамках одного диалога."""

    budget_min: Optional[int] = Field(default=None, ge=0)
    budget_max: Optional[int] = Field(default=None, ge=0)

    purpose: Optional[str] = None  # Цель покупки.
    experience_level: Optional[str] = None  # Опыт водителя.

    family_size: Optional[int] = Field(default=None, ge=1)

    preferred_body_types: List[str] = Field(default_factory=list)
    preferred_brands: List[str] = Field(default_factory=list)

    must_have: List[str] = Field(default_factory=list)  # То, что желательно проверять как важное условие.
    must_not_have: List[str] = Field(default_factory=list)  # То, что нельзя предлагать.

    user_notes: str = Field(default="")  # Нюансы, которые пока не вынесены в отдельные поля.

    selected_car_id: Optional[UUID] = None  # Конкретная выбранная машина.

    # Текущий статус диалога, агент начинает с первичного опроса.
    dialog_status: DialogStatus = DialogStatus.INITIAL_SURVEY

    def has_required_data_for_recommendation(self) -> bool:
        """
        Минимум:
        1. Верхняя граница бюджета.
        2. Цель покупки.
        """
        return self.budget_max is not None and self.purpose is not None

    def has_selected_car(self) -> bool:
        """Проверяет, выбрал ли пользователь конкретную машину."""
        return self.selected_car_id is not None

    def mark_ready_to_recommend(self) -> None:
        """Переводит диалог в состояние готовности к рекомендации."""
        if self.has_required_data_for_recommendation():
            self.dialog_status = DialogStatus.READY_TO_RECOMMEND

    def select_car(self, car_id: UUID) -> None:
        """Сохраняет выбранную пользователем машину."""
        self.selected_car_id = car_id
        self.dialog_status = DialogStatus.CAR_SELECTED

    def mark_ready_for_reservation(self) -> None:
        """Переводит диалог к созданию заявки."""
        if self.has_selected_car():
            self.dialog_status = DialogStatus.READY_FOR_RESERVATION

    def mark_reservation_created(self) -> None:
        """Фиксирует, что mock-заявка создана."""
        self.dialog_status = DialogStatus.RESERVATION_CREATED