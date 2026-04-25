from typing import Optional

from pydantic import BaseModel, Field

from app.domain.agent_response import SessionUpdate


class ExtractorResult(BaseModel):
    """Результат извлечения требований из сообщения пользователя."""

    session_update: SessionUpdate = Field(
        ...,
        description="Данные, которые нужно записать в память диалога.",
    )

    selected_car_title: Optional[str] = Field(
        default=None,
        description="Название машины, если пользователь выбрал ее текстом. Например: Toyota Corolla.",
    )

    should_ask_clarifying_question: bool = Field(
        default=False,
        description="True, если не хватает данных и пользователю нужно задать уточняющий вопрос.",
    )

    clarifying_question: Optional[str] = Field(
        default=None,
        description="Текст уточняющего вопроса.",
    )

    user_message: str = Field(
        ...,
        description="Краткий черновик ответа пользователю.",
    )