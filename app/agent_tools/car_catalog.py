import json
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from pydantic import ValidationError

from app.domain.car import Car


class CarCatalog:

    def __init__(self, json_path: str = "../data/cars.json") -> None:
        self.json_path = Path(json_path)
        self._cars: Optional[List[Car]] = None  # Кэш каталога.

    def validate_catalog(self) -> bool:
        """Проверяет, что JSON-каталог валиден."""
        self._load_cars()
        return True

    def _load_cars(self) -> List[Car]:
        """Читает JSON-файл, валидирует машины и сохраняет их в память."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Файл с машинами не найден: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Каталог машин должен быть JSON-массивом.")

        try:
            cars = [Car(**raw_car) for raw_car in data]
        except ValidationError as error:
            raise ValueError(f"Ошибка валидации каталога машин: {error}") from error

        self._cars = cars
        return cars

    def _get_cars(self) -> List[Car]:
        """Возвращает машины из памяти или загружает их при первом обращении."""
        if self._cars is None:
            return self._load_cars()

        return self._cars

    def find_all(self) -> List[Car]:
        """Возвращает все машины из каталога."""
        return self._get_cars()

    def find_by_id(self, car_id: UUID) -> Optional[Car]:
        """Ищет машину по UUID."""
        for car in self._get_cars():
            if car.id == car_id:
                return car

        return None

    def find_by_budget(
            self,
            budget_min: Optional[int],
            budget_max: Optional[int],
    ) -> List[Car]:
        """Ищет машины, которые попадают в бюджет."""
        result: List[Car] = []

        for car in self._get_cars():
            if car.fits_budget(budget_min=budget_min, budget_max=budget_max):
                result.append(car)

        return result

    def find_by_filters(
            self,
            budget_min: Optional[int] = None,
            budget_max: Optional[int] = None,
            body_type: Optional[str] = None,
            fuel_type: Optional[str] = None,
            transmission: Optional[str] = None,
            drive_type: Optional[str] = None,
            brand: Optional[str] = None,
    ) -> List[Car]:
        """Ищет машины по основным фильтрам."""
        result: List[Car] = []

        for car in self._get_cars():
            if not car.fits_budget(budget_min=budget_min, budget_max=budget_max):
                continue

            if body_type is not None and car.body_type.value != body_type.lower():
                continue

            if fuel_type is not None and car.fuel_type.value != fuel_type.lower():
                continue

            if transmission is not None and car.transmission.value != transmission.lower():
                continue

            if drive_type is not None and car.drive_type.value != drive_type.lower():
                continue

            if brand is not None and car.brand.lower() != brand.lower():
                continue

            result.append(car)

        return result
