from typing import List, Optional

from pydantic import BaseModel, Field


class SessionUpdate(BaseModel):
    """Данные, которые ExtractorAgent извлек из сообщения пользователя."""

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


class ExtractorResult(BaseModel):
    """Результат извлечения требований из сообщения пользователя."""

    session_update: SessionUpdate = Field(
        ...,
        description=(
            "Данные, которые нужно записать в память диалога. "
            "Объект обязателен. Внутри него все поля тоже обязательны: "
            "если значение неизвестно, верни null, [], или ''."
        ),
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
