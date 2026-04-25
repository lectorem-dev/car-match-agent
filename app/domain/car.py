from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Тип кузова.
class BodyType(str, Enum):
    SEDAN = "sedan"
    HATCHBACK = "hatchback"
    LIFTBACK = "liftback"
    WAGON = "wagon"
    SUV = "suv"
    COUPE = "coupe"
    MINIVAN = "minivan"
    PICKUP = "pickup"

# Тип топлива.
class FuelType(str, Enum):
    PETROL = "petrol"
    DIESEL = "diesel"
    HYBRID = "hybrid"
    ELECTRIC = "electric"


# Тип коробки передач.
class Transmission(str, Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    ROBOT = "robot"
    CVT = "cvt"


# Тип привода.
class DriveType(str, Enum):
    FWD = "fwd"
    RWD = "rwd"
    AWD = "awd"


class Car(BaseModel):
    id: UUID
    brand: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    year: int = Field(..., ge=1980)
    price: int = Field(..., gt=0, description="Цена в долларах") # Цена в долларах.
    body_type: BodyType
    mileage_km: int = Field(..., ge=0) # Пробег автомобиля
    fuel_type: FuelType
    transmission: Transmission
    drive_type: DriveType
    engine_power_hp: int = Field(..., gt=0) # Мощность двигателя в лошадиных силах.
    description: str = Field(default="") # Свободное описание машины.

    def title(self) -> str:
        """Возвращает короткое название машины для отображения в ответах."""
        return f"{self.brand} {self.model} {self.year}"

    def fits_budget(
            self,
            budget_min: Optional[int],
            budget_max: Optional[int],
    ) -> bool:
        """ Проверяет, попадает ли машина в бюджет пользователя."""
        if budget_min is not None and self.price < budget_min:
            return False

        if budget_max is not None and self.price > budget_max:
            return False

        return True