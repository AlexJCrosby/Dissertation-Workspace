from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, messagebox

from assistant import TaskManager, parse_intent, handle_command, Intent
from asr.whisper_asr.whisper_integration import transcribe_microphone


class VoiceTaskAssistantGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Voice Task Assistant")
        self.root.geometry("900x650")

        self.task_manager = TaskManager(storage_path="tasks.json")
        self.is_recording = False

        self.transcript_var = tk.StringVar(value="Transcript will appear here.")
        self.intent_var = tk.StringVar(value="Intent: -")
        self.status_var = tk.StringVar(value="Ready.")
        self.manual_command_var = tk.StringVar()

        self._build_ui()
        self.refresh_tasks()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top_frame = ttk.Frame(self.root, padding=12)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(0, weight=1)

        title = ttk.Label(top_frame, text="Voice Task Assistant", font=("Segoe UI", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            top_frame,
            text="Record a voice command, or type one manually for testing.",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        main_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        main_frame.grid(row=1, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)

        command_frame = ttk.LabelFrame(main_frame, text="Command Input", padding=12)
        command_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        command_frame.columnconfigure(0, weight=1)

        self.record_button = ttk.Button(command_frame, text="Record Voice Command", command=self.start_voice_command)
        self.record_button.grid(row=0, column=0, sticky="w")

        ttk.Label(command_frame, text="Manual command:").grid(row=1, column=0, sticky="w", pady=(12, 4))

        manual_row = ttk.Frame(command_frame)
        manual_row.grid(row=2, column=0, sticky="ew")
        manual_row.columnconfigure(0, weight=1)

        self.manual_entry = ttk.Entry(manual_row, textvariable=self.manual_command_var)
        self.manual_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.manual_entry.bind("<Return>", lambda event: self.run_manual_command())

        submit_button = ttk.Button(manual_row, text="Run Command", command=self.run_manual_command)
        submit_button.grid(row=0, column=1, sticky="e")

        transcript_frame = ttk.LabelFrame(main_frame, text="Latest Recognition", padding=12)
        transcript_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        transcript_frame.columnconfigure(0, weight=1)
        transcript_frame.rowconfigure(1, weight=1)

        ttk.Label(transcript_frame, text="Transcript:").grid(row=0, column=0, sticky="w")
        self.transcript_label = ttk.Label(
            transcript_frame,
            textvariable=self.transcript_var,
            wraplength=500,
            justify="left",
        )
        self.transcript_label.grid(row=1, column=0, sticky="nw", pady=(4, 12))

        self.intent_label = ttk.Label(transcript_frame, textvariable=self.intent_var)
        self.intent_label.grid(row=2, column=0, sticky="w")

        self.help_text = tk.Text(transcript_frame, height=10, wrap="word")
        self.help_text.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.help_text.insert(
            "1.0",
            "Examples:\n"
            "- add task wash dishes\n"
            "- complete task wash dishes\n"
            "- delete task wash dishes\n"
            "- list tasks\n"
            "- help\n"
        )
        self.help_text.configure(state="disabled")

        tasks_frame = ttk.LabelFrame(main_frame, text="Tasks", padding=12)
        tasks_frame.grid(row=1, column=1, sticky="nsew")
        tasks_frame.columnconfigure(0, weight=1)
        tasks_frame.rowconfigure(0, weight=1)

        columns = ("display_index", "task_id", "status", "text")
        self.tree = ttk.Treeview(tasks_frame, columns=columns, show="headings", height=15)
        self.tree.heading("display_index", text="#")
        self.tree.heading("task_id", text="ID")
        self.tree.heading("status", text="Status")
        self.tree.heading("text", text="Task")

        self.tree.column("display_index", width=40, anchor="center")
        self.tree.column("task_id", width=50, anchor="center")
        self.tree.column("status", width=80, anchor="center")
        self.tree.column("text", width=260, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(tasks_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        actions = ttk.Frame(tasks_frame)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        actions.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(actions, text="Refresh", command=self.refresh_tasks).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(actions, text="Complete Selected", command=self.complete_selected_task).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(actions, text="Delete Selected", command=self.delete_selected_task).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken", padding=(8, 4))
        status_bar.grid(row=2, column=0, sticky="ew")

    def refresh_tasks(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        tasks = self.task_manager.list_tasks()
        for display_index, task in enumerate(tasks, start=1):
            status = "Complete" if task.completed else "Open"
            self.tree.insert(
                "",
                "end",
                values=(display_index, task.id, status, task.text),
            )

        self.status_var.set(f"Loaded {len(tasks)} task(s).")

    def start_voice_command(self) -> None:
        if self.is_recording:
            return

        self.is_recording = True
        self.record_button.configure(state="disabled")
        self.status_var.set("Recording and transcribing...")
        self.transcript_var.set("Listening...")
        self.intent_var.set("Intent: -")

        thread = threading.Thread(target=self._record_and_process, daemon=True)
        thread.start()

    def _record_and_process(self) -> None:
        try:
            transcript, wav_path = transcribe_microphone(duration=3.0)
            command = parse_intent(transcript)
            self.root.after(0, self._apply_command_result, transcript, command.intent.value)
            self.root.after(0, self._execute_parsed_command, command)
        except Exception as exc:
            self.root.after(0, self._handle_error, str(exc))
        finally:
            self.root.after(0, self._reset_record_button)

    def _execute_parsed_command(self, command) -> None:
        self.intent_var.set(f"Intent: {command.intent.value}")

        if command.intent == Intent.HELP:
            self.status_var.set("Help shown in examples panel.")
            return

        if command.intent == Intent.LIST_TASKS:
            self.refresh_tasks()
            self.status_var.set("Listed tasks.")
            return

        if command.intent == Intent.ADD_TASK:
            if not command.task_text:
                self.status_var.set("Add task failed: no task text detected.")
                return
            task = self.task_manager.add_task(command.task_text)
            self.refresh_tasks()
            self.status_var.set(f'Added task #{task.id}: "{task.text}"')
            return

        if command.intent == Intent.COMPLETE_TASK:
            if not command.task_text:
                self.status_var.set("Complete task failed: no task detected.")
                return
            task = self.task_manager.complete_task(command.task_text)
            self.refresh_tasks()
            if task is None:
                self.status_var.set(f'No task matched: "{command.task_text}"')
            else:
                self.status_var.set(f'Marked task #{task.id} complete: "{task.text}"')
            return

        if command.intent == Intent.DELETE_TASK:
            if not command.task_text:
                self.status_var.set("Delete task failed: no task detected.")
                return
            task = self.task_manager.delete_task(command.task_text)
            self.refresh_tasks()
            if task is None:
                self.status_var.set(f'No task matched: "{command.task_text}"')
            else:
                self.status_var.set(f'Deleted task #{task.id}: "{task.text}"')
            return

        self.status_var.set("Sorry, I did not understand that command.")

    def _apply_command_result(self, transcript: str, intent_text: str) -> None:
        self.transcript_var.set(transcript if transcript else "<empty transcript>")
        self.intent_var.set(f"Intent: {intent_text}")

    def _handle_error(self, error_text: str) -> None:
        self.transcript_var.set("<error>")
        self.intent_var.set("Intent: -")
        self.status_var.set(f"Recording/transcription failed: {error_text}")
        messagebox.showerror("Voice Task Assistant", error_text)

    def _reset_record_button(self) -> None:
        self.is_recording = False
        self.record_button.configure(state="normal")

    def run_manual_command(self) -> None:
        transcript = self.manual_command_var.get().strip()
        if not transcript:
            self.status_var.set("Enter a manual command first.")
            return

        command = parse_intent(transcript)
        self._apply_command_result(transcript, command.intent.value)
        self._execute_parsed_command(command)
        self.manual_command_var.set("")

    def _get_selected_task_id(self) -> int | None:
        selection = self.tree.selection()
        if not selection:
            self.status_var.set("Select a task first.")
            return None

        item = self.tree.item(selection[0])
        values = item.get("values", [])
        if len(values) < 2:
            self.status_var.set("Could not read selected task.")
            return None

        try:
            return int(values[1])
        except (TypeError, ValueError):
            self.status_var.set("Invalid task ID in selection.")
            return None

    def complete_selected_task(self) -> None:
        task_id = self._get_selected_task_id()
        if task_id is None:
            return

        task = self.task_manager.complete_task(str(task_id))
        self.refresh_tasks()
        if task is None:
            self.status_var.set(f"Could not complete task ID {task_id}.")
        else:
            self.status_var.set(f'Marked task #{task.id} complete: "{task.text}"')

    def delete_selected_task(self) -> None:
        task_id = self._get_selected_task_id()
        if task_id is None:
            return

        task = self.task_manager.delete_task(str(task_id))
        self.refresh_tasks()
        if task is None:
            self.status_var.set(f"Could not delete task ID {task_id}.")
        else:
            self.status_var.set(f'Deleted task #{task.id}: "{task.text}"')


def main() -> None:
    root = tk.Tk()
    app = VoiceTaskAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
