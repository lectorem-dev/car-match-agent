from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.critic.agent import CriticAgent
from app.agents.guard.agent import DomainGuardAgent
from app.agents.extractor.agent import Extractor
from app.agents.planner.agent import PlannerAgent
from app.agents.reservation.agent import ReservationAgent
from app.catalog.car_catalog import CarCatalog
from app.evals.loader import ScenarioLoader
from app.evals.runner import ScenarioRunner
from app.llm.yandex_llm_client import YandexLLMClient
from app.orchestrator.pipeline import Pipeline
from app.services.session_update_service import SessionUpdateService


# Флаг для вывода технических логов LLM-клиента.
ENABLE_LLM_LOGS = True

# Флаг для вывода логов пайплайна.
ENABLE_PIPELINE_LOGS = True


def build_pipeline(project_root: Path) -> Pipeline:
    """Собирает агентный пайплайн и зависимости."""

    catalog_path = project_root / "data" / "cars-18.json"

    catalog = CarCatalog(json_path=str(catalog_path))
    catalog.validate_catalog()

    llm_client = YandexLLMClient(
        enable_logging=ENABLE_LLM_LOGS,
    )

    session_update_service = SessionUpdateService()

    domain_guard = DomainGuardAgent(
        llm_client=llm_client,
    )

    reservation = ReservationAgent(
        llm_client=llm_client,
        catalog=catalog,
    )

    extractor = Extractor(
        llm_client=llm_client,
        session_update_service=session_update_service,
    )

    planner = PlannerAgent(
        llm_client=llm_client,
        catalog_tool=catalog,
    )

    critic = CriticAgent(
        llm_client=llm_client,
    )

    pipeline = Pipeline(
        domain_guard=domain_guard,
        reservation=reservation,
        extractor=extractor,
        planner=planner,
        critic=critic,
        catalog=catalog,
        session_update_service=session_update_service,
        enable_logging=ENABLE_PIPELINE_LOGS,
    )

    return pipeline


def run_eval_suite(project_root: Path) -> None:
    """Запускает тестовый набор сценариев."""

    print("Этап 1. Инициализация агента")

    pipeline = build_pipeline(project_root=project_root)

    scenarios_path = project_root / "evals" / "scenarios-10.json"
    loader = ScenarioLoader(json_path=str(scenarios_path))
    scenarios = loader.load()

    print(f"Сценарии загружены: {len(scenarios)}")
    print("Этап 2. Запуск eval-набора\n")

    runner = ScenarioRunner(agent=pipeline)

    passed_count = 0

    for scenario in scenarios:
        result = runner.run(scenario)

        if result.passed:
            passed_count += 1
            print(f"[OK] {result.scenario_id}: {result.scenario_name}")
        else:
            print(f"[FAIL] {result.scenario_id}: {result.scenario_name}")
            print(f"Причина: {result.error}")

        print("-" * 80)

    print("\nИтог eval-набора")
    print(f"Пройдено: {passed_count}/{len(scenarios)}")


def main() -> None:
    """Точка входа приложения."""

    project_root = Path(__file__).resolve().parent.parent

    run_eval_suite(project_root=project_root)


if __name__ == "__main__":
    main()
