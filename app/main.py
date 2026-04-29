from pathlib import Path
from contextlib import contextmanager
import re
import sys
from typing import Iterator, Optional, TextIO


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.critic.critic_agent import CriticAgent
from app.agents.guard.guard_agent import DomainGuardAgent
from app.agents.extractor.extractor_agent import Extractor
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.reservation.reservation_agent import ReservationAgent
from app.agent_tools.car_catalog import CarCatalog
from app.evals.eval_loader import ScenarioLoader
from app.evals.eval_runner import ScenarioRunner
from app.interactive.console_chat import run_interactive_chat
from app.llm.yandex_llm_client import YandexLLMClient
from app.orchestrator.conversation_pipeline import Pipeline
from app.session.session_update_service import SessionUpdateService


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATALOG_JSON_PATH = PROJECT_ROOT    / "data"    / "cars.json"
SCENARIOS_JSON_PATH = PROJECT_ROOT  / "evals"   / "scenarios_full.json"
CONSOLE_LOG_PATH = PROJECT_ROOT     / "output"  / "eval_logs.txt"

ENABLE_LLM_LOGS = False                    # Флаг для вывода технических логов LLM-клиента.
ENABLE_PIPELINE_LOGS = False                # Флаг для вывода логов пайплайна.
ENABLE_CONSOLE_LOG_FILE = False             # Флаг для сохранения вывода консоли в txt файл

ENABLE_DOMAIN_GUARD_AGENT_LOGS = False      # Флаг для вывода логов DomainGuardAgent.
ENABLE_RESERVATION_AGENT_LOGS = False       # Флаг для вывода логов ReservationAgent.
ENABLE_EXTRACTOR_AGENT_LOGS = False         # Флаг для вывода логов ExtractorAgent.
ENABLE_PLANNER_AGENT_LOGS = False           # Флаг для вывода логов PlannerAgent.
ENABLE_CRITIC_AGENT_LOGS = False            # Флаг для вывода логов CriticAgent.

EVAL_MODE = False                           # Флаг запуска eval-сценариев.
INTERACTIVE_MODE = True                   # Флаг запуска интерактивного чата в консоли.


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


def build_pipeline() -> Pipeline:
    """Собирает агентный пайплайн и зависимости."""

    catalog = CarCatalog(json_path=str(CATALOG_JSON_PATH))
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


def run_eval_suite(pipeline: Optional[Pipeline] = None) -> None:
    """Запускает тестовый набор сценариев."""

    print("Этап 1. Инициализация агента")

    if pipeline is None:
        pipeline = build_pipeline()

    loader = ScenarioLoader(json_path=str(SCENARIOS_JSON_PATH))
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

    with duplicate_console_output(
            log_path=CONSOLE_LOG_PATH,
            enabled=ENABLE_CONSOLE_LOG_FILE,
    ):
        pipeline = build_pipeline() if (EVAL_MODE or INTERACTIVE_MODE) else None

        if EVAL_MODE:
            run_eval_suite(pipeline=pipeline)

        if INTERACTIVE_MODE:
            run_interactive_chat(pipeline=pipeline)


if __name__ == "__main__":
    main()
