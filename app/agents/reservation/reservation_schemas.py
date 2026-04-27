from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# Статус намерения бронирования.
class ReservationIntent(str, Enum):
    RESERVATION_REQUEST = "reservation_request"
    NOT_RESERVATION_REQUEST = "not_reservation_request"


class ReservationResult(BaseModel):
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


class ReservationCreatedResult(BaseModel):
    """Результат успешного создания mock-заявки."""

    user_message: str
    ready_for_reservation: bool = True
