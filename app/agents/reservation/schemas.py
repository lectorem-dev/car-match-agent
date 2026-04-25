from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Статус намерения бронирования.
class ReservationIntent(str, Enum):
    RESERVATION_REQUEST = "reservation_request"
    NOT_RESERVATION_REQUEST = "not_reservation_request"


# Результат обработки бронирования.
class ReservationStatus(str, Enum):
    CREATED = "created"
    NEED_CAR_SELECTION = "need_car_selection"
    CAR_NOT_FOUND = "car_not_found"
    NOT_RESERVATION = "not_reservation"


class ReservationDecision(BaseModel):
    """Решение ReservationAgent по сообщению пользователя."""

    intent: ReservationIntent = Field(
        ...,
        description="Просит ли пользователь забронировать автомобиль.",
    )

    selected_car_title: Optional[str] = Field(
        default=None,
        description="Название машины, если пользователь указал ее в сообщении.",
    )

    user_message: str = Field(
        ...,
        description="Краткий ответ пользователю.",
    )


class ReservationResult(BaseModel):
    """Результат работы ReservationAgent."""

    status: ReservationStatus
    user_message: str
    selected_car_id: Optional[UUID] = None