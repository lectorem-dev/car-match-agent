from pathlib import Path
from contextlib import contextmanager
import re
import sys
from typing import Iterator, TextIO


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



ENABLE_LLM_LOGS = False                    # Флаг для вывода технических логов LLM-клиента.
ENABLE_PIPELINE_LOGS = True                # Флаг для вывода логов пайплайна.
ENABLE_CONSOLE_LOG_FILE = True            # Флаг для сохранения вывода консоли в output/logs.txt.

ENABLE_DOMAIN_GUARD_AGENT_LOGS = True      # Флаг для вывода логов DomainGuardAgent.
ENABLE_RESERVATION_AGENT_LOGS = True       # Флаг для вывода логов ReservationAgent.
ENABLE_EXTRACTOR_AGENT_LOGS = True         # Флаг для вывода логов ExtractorAgent.
ENABLE_PLANNER_AGENT_LOGS = True           # Флаг для вывода логов PlannerAgent.
ENABLE_CRITIC_AGENT_LOGS = True            # Флаг для вывода логов CriticAgent.


class TeeStream:
    """Дублирует вывод в консоль и в файл."""

    ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, console_stream: TextIO, file_stream: TextIO) -> None:
        self.console_stream = console_stream
        self.file_stream = file_stream

    def write(self, data: str) -> int:
        self.console_stream.write(data)
        self.file_stream.write(self.ANSI_ESCAPE_RE.sub("", data))
        return len(data)

    def flush(self) -> None:
        self.console_stream.flush()
        self.file_stream.flush()

    def isatty(self) -> bool:
        return self.console_stream.isatty()

    @property
    def encoding(self) -> str:
        return self.console_stream.encoding

    def __getattr__(self, name: str):
        return getattr(self.console_stream, name)


@contextmanager
def duplicate_console_output(log_path: Path, enabled: bool) -> Iterator[None]:
    """По флагу дублирует stdout/stderr в лог-файл."""

    if not enabled:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)

        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


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
        enable_logs=ENABLE_DOMAIN_GUARD_AGENT_LOGS,
    )

    reservation = ReservationAgent(
        llm_client=llm_client,
        catalog=catalog,
        enable_logs=ENABLE_RESERVATION_AGENT_LOGS,
    )

    extractor = Extractor(
        llm_client=llm_client,
        session_update_service=session_update_service,
        enable_logs=ENABLE_EXTRACTOR_AGENT_LOGS,
    )

    planner = PlannerAgent(
        llm_client=llm_client,
        catalog_tool=catalog,
        enable_logs=ENABLE_PLANNER_AGENT_LOGS,
    )

    critic = CriticAgent(
        llm_client=llm_client,
        enable_logs=ENABLE_CRITIC_AGENT_LOGS,
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

    runner = ScenarioRunner(
        agent=pipeline,
        enable_logs=ENABLE_PIPELINE_LOGS,
    )

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
    logs_path = project_root / "output" / "logs.txt"

    with duplicate_console_output(
            log_path=logs_path,
            enabled=ENABLE_CONSOLE_LOG_FILE,
    ):
        run_eval_suite(project_root=project_root)


if __name__ == "__main__":
    main()
