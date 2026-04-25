import json
from pathlib import Path
from typing import List

from pydantic import ValidationError

from app.evals.schemas import TestScenario


class ScenarioLoader:
    """Читает и валидирует тестовые сценарии."""

    def __init__(self, json_path: str = "app/evals/scenarios.json") -> None:
        self.json_path = Path(json_path)

    def load(self) -> List[TestScenario]:
        """Загружает сценарии из JSON-файла."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Файл сценариев не найден: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Файл сценариев должен быть JSON-массивом.")

        try:
            return [TestScenario(**item) for item in data]
        except ValidationError as error:
            raise ValueError(f"Ошибка валидации сценариев: {error}") from error
