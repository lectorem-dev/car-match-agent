from typing import Optional

from app.domain.user_session import DialogStatus, UserSession
from app.orchestrator.conversation_pipeline import Pipeline
from app.orchestrator.orchestrator_schemas import PipelineResponse


EXIT_COMMANDS = {"exit", "/exit"}
RESTART_COMMANDS = {"restart", "/restart"}
INTERACTIVE_SCENARIO_NAME = "interactive"


def run_interactive_chat(pipeline: Optional[Pipeline] = None) -> None:
    """Запускает интерактивный консольный чат с агентом."""

    if pipeline is None:
        from app.main import build_pipeline

        pipeline = build_pipeline()

    session = UserSession()
    session_number = 1

    print("Агент подбора автомобилей. Интерактивный режим.")
    _print_session_banner(session_number=session_number)

    while True:
        try:
            user_message = input("Покупатель: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Интерактивный режим завершен.")
            break

        normalized_message = user_message.lower()

        if normalized_message in EXIT_COMMANDS:
            print()
            print("Интерактивный режим завершен.")
            break

        if normalized_message in RESTART_COMMANDS:
            session = UserSession()
            session_number += 1
            _print_session_banner(session_number=session_number)
            continue

        if not user_message:
            print()
            continue

        try:
            response = pipeline.handle_message(
                user_message=user_message,
                session=session,
                allow_clarifying_question=True,
                scenario_name=INTERACTIVE_SCENARIO_NAME,
            )
        except Exception:
            print("Агент: Не удалось обработать сообщение. Проверьте логи.")
            print()
            continue

        print(f"Агент: {_render_agent_message(response)}")
        print()

        if session.dialog_status == DialogStatus.RESERVATION_CREATED:
            session = UserSession()
            session_number += 1
            _print_session_banner(session_number=session_number)
            continue


def _print_session_banner(session_number: int) -> None:
    """Печатает заголовок новой интерактивной сессии."""

    print()
    print("=" * 60)
    print(f"Сессия #{session_number}")
    print("-" * 60)
    print("Доступные команды:")
    print("  exit или /exit       - выход")
    print("  restart или /restart - перезапустить диалог")
    print("=" * 60)
    print()


def _render_agent_message(response: PipelineResponse) -> str:
    """Возвращает пользовательский текст ответа без служебных структур."""

    if response.user_message.strip():
        return response.user_message

    if response.should_ask_clarifying_question and response.clarifying_question:
        return response.clarifying_question

    if response.ready_for_reservation:
        return "Mock-заявка создана."

    if response.recommended_cars:
        return "Подобрал варианты."

    return "Не удалось сформировать ответ."
