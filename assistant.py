from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

from asr.whisper_asr.whisper_integration import transcribe_microphone


class Intent(str, Enum):
    ADD_TASK = "add_task"
    COMPLETE_TASK = "complete_task"
    DELETE_TASK = "delete_task"
    LIST_TASKS = "list_tasks"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class Task:
    id: int
    text: str
    completed: bool = False


@dataclass
class ParsedCommand:
    intent: Intent
    task_text: Optional[str] = None
    raw_text: str = ""


class TaskManager:
    def __init__(self) -> None:
        self.tasks: list[Task] = []
        self.next_id: int = 1

    def add_task(self, text: str) -> Task:
        text = self._clean_task_text(text)
        task = Task(id=self.next_id, text=text, completed=False)
        self.tasks.append(task)
        self.next_id += 1
        return task

    def complete_task(self, query: str) -> Optional[Task]:
        task = self._find_task(query)
        if task is None:
            return None
        task.completed = True
        return task

    def delete_task(self, query: str) -> Optional[Task]:
        task = self._find_task(query)
        if task is None:
            return None
        self.tasks.remove(task)
        return task

    def list_tasks(self) -> list[Task]:
        return self.tasks

    def _find_task(self, query: str) -> Optional[Task]:
        cleaned = self._clean_task_text(query)
        if not cleaned:
            return None

        # 1) Exact ID match, e.g. "2" or "task 2"
        id_match = re.search(r"\b(?:task\s*)?(\d+)\b", cleaned)
        if id_match:
            task_id = int(id_match.group(1))
            for task in self.tasks:
                if task.id == task_id:
                    return task

        # 2) Exact text match
        for task in self.tasks:
            if task.text.lower() == cleaned.lower():
                return task

        # 3) Substring match
        for task in self.tasks:
            if cleaned.lower() in task.text.lower():
                return task

        return None

    @staticmethod
    def _clean_task_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^[\-:,. ]+", "", text)
        text = re.sub(r"[\-:,. ]+$", "", text)
        return text


def normalise_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def strip_prefix(text: str, prefixes: list[str]) -> str:
    for prefix in sorted(prefixes, key=len, reverse=True):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text.strip()


def parse_intent(transcript: str) -> ParsedCommand:
    raw_text = transcript.strip()
    text = normalise_text(transcript)

    if not text:
        return ParsedCommand(intent=Intent.UNKNOWN, raw_text=raw_text)

    help_phrases = [
        "help",
        "ask for help",
        "what can i say",
        "what can you do",
        "show help",
    ]
    list_phrases = [
        "list tasks",
        "show tasks",
        "read tasks",
        "what are my tasks",
        "what tasks do i have",
        "show my tasks",
        "list my tasks",
    ]
    add_prefixes = [
        "add task",
        "create task",
        "new task",
        "add",
    ]
    complete_prefixes = [
        "task complete",
        "complete task",
        "mark task complete",
        "mark complete",
        "finish task",
        "complete",
        "done",
    ]
    delete_prefixes = [
        "delete task",
        "remove task",
        "delete",
        "remove",
    ]

    if text in help_phrases or any(text.startswith(p) for p in help_phrases):
        return ParsedCommand(intent=Intent.HELP, raw_text=raw_text)

    if text in list_phrases or any(text.startswith(p) for p in list_phrases):
        return ParsedCommand(intent=Intent.LIST_TASKS, raw_text=raw_text)

    if any(text.startswith(prefix) for prefix in add_prefixes):
        task_text = strip_prefix(text, add_prefixes)
        return ParsedCommand(intent=Intent.ADD_TASK, task_text=task_text, raw_text=raw_text)

    if any(text.startswith(prefix) for prefix in complete_prefixes):
        task_text = strip_prefix(text, complete_prefixes)
        return ParsedCommand(intent=Intent.COMPLETE_TASK, task_text=task_text, raw_text=raw_text)

    if any(text.startswith(prefix) for prefix in delete_prefixes):
        task_text = strip_prefix(text, delete_prefixes)
        return ParsedCommand(intent=Intent.DELETE_TASK, task_text=task_text, raw_text=raw_text)

    return ParsedCommand(intent=Intent.UNKNOWN, raw_text=raw_text)


def format_tasks(tasks: list[Task]) -> str:
    if not tasks:
        return "No tasks yet."

    lines = []
    for task in tasks:
        status = "[x]" if task.completed else "[ ]"
        lines.append(f"{task.id}. {status} {task.text}")
    return "\n".join(lines)


def print_help() -> None:
    print("\nAvailable voice commands:")
    print("- add task wash dishes")
    print("- complete task wash dishes")
    print("- delete task wash dishes")
    print("- list tasks")
    print("- help")


def handle_command(command: ParsedCommand, task_manager: TaskManager) -> None:
    print(f"\nIntent: {command.intent.value}")

    if command.intent == Intent.HELP:
        print_help()
        return

    if command.intent == Intent.LIST_TASKS:
        print("\nCurrent tasks:")
        print(format_tasks(task_manager.list_tasks()))
        return

    if command.intent == Intent.ADD_TASK:
        if not command.task_text:
            print("I heard an add-task command, but no task text was detected.")
            return
        task = task_manager.add_task(command.task_text)
        print(f'Added task #{task.id}: "{task.text}"')
        print("\nCurrent tasks:")
        print(format_tasks(task_manager.list_tasks()))
        return

    if command.intent == Intent.COMPLETE_TASK:
        if not command.task_text:
            print("I heard a complete-task command, but no task was detected.")
            return
        task = task_manager.complete_task(command.task_text)
        if task is None:
            print(f'Could not find a task matching: "{command.task_text}"')
        else:
            print(f'Marked task #{task.id} complete: "{task.text}"')
            print("\nCurrent tasks:")
            print(format_tasks(task_manager.list_tasks()))
        return

    if command.intent == Intent.DELETE_TASK:
        if not command.task_text:
            print("I heard a delete-task command, but no task was detected.")
            return
        task = task_manager.delete_task(command.task_text)
        if task is None:
            print(f'Could not find a task matching: "{command.task_text}"')
        else:
            print(f'Deleted task #{task.id}: "{task.text}"')
            print("\nCurrent tasks:")
            print(format_tasks(task_manager.list_tasks()))
        return

    print("Sorry, I did not understand that command.")
    print_help()


def main() -> None:
    task_manager = TaskManager()

    print("Voice Task Assistant")
    print("Press Enter to record a command, or type 'q' to quit.")
    print_help()

    while True:
        user_input = input("\n[Enter] record | [q] quit: ").strip().lower()
        if user_input == "q":
            print("Exiting assistant.")
            break

        try:
            transcript, wav_path = transcribe_microphone(duration=3.0)
        except Exception as exc:
            print(f"\nRecording/transcription failed: {exc}")
            continue

        print(f"\nWAV: {wav_path}")
        print(f"Transcript: {transcript}")

        command = parse_intent(transcript)
        handle_command(command, task_manager)


if __name__ == "__main__":
    main()
