from enum import Enum

from pydantic import BaseModel, Field


# Статус домена запроса.
class DomainStatus(str, Enum):
    IN_DOMAIN = "in_domain"
    OUT_OF_DOMAIN = "out_of_domain"


class DomainGuardResult(BaseModel):
    """Результат проверки домена запроса."""

    domain_status: DomainStatus = Field(
        ...,
        description="in_domain, если запрос относится к автомобилям; out_of_domain, если нет.",
    )

    reason: str = Field(
        ...,
        description="Краткое объяснение решения.",
    )

    user_message: str = Field(
        ...,
        description="Ответ пользователю, если запрос вне домена.",
    )