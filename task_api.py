from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request

from assistant_vosk import TaskManager

app = Flask(__name__)

TASKS_PATH = Path("tasks.json")
task_manager = TaskManager(storage_path=TASKS_PATH)
task_lock = Lock()


def serialise_tasks():
    return [asdict(task) for task in task_manager.list_tasks()]


@app.get("/health")
def health():
    return jsonify({"ok": True, "message": "Task API running"}), 200


@app.get("/tasks")
def get_tasks():
    with task_lock:
        return jsonify({
            "ok": True,
            "tasks": serialise_tasks()
        }), 200


@app.post("/tasks")
def add_task():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({
            "ok": False,
            "message": "Task text is required"
        }), 400

    with task_lock:
        task = task_manager.add_task(text)
        return jsonify({
            "ok": True,
            "message": "Task added",
            "task": asdict(task),
            "tasks": serialise_tasks()
        }), 201


@app.post("/tasks/complete")
def complete_tasks():
    data = request.get_json(silent=True) or {}
    ids = data.get("ids", [])

    if not isinstance(ids, list) or not ids:
        return jsonify({
            "ok": False,
            "message": "A non-empty list of task IDs is required"
        }), 400

    completed_ids = []
    not_found_ids = []

    with task_lock:
        for task_id in ids:
            matched = task_manager.complete_task(str(task_id))
            if matched is None:
                not_found_ids.append(task_id)
            else:
                completed_ids.append(task_id)

        return jsonify({
            "ok": True,
            "message": "Complete request processed",
            "completed_ids": completed_ids,
            "not_found_ids": not_found_ids,
            "tasks": serialise_tasks()
        }), 200


@app.post("/tasks/delete")
def delete_tasks():
    data = request.get_json(silent=True) or {}
    ids = data.get("ids", [])

    if not isinstance(ids, list) or not ids:
        return jsonify({
            "ok": False,
            "message": "A non-empty list of task IDs is required"
        }), 400

    deleted_ids = []
    not_found_ids = []

    with task_lock:
        for task_id in ids:
            matched = task_manager.delete_task(str(task_id))
            if matched is None:
                not_found_ids.append(task_id)
            else:
                deleted_ids.append(task_id)

        return jsonify({
            "ok": True,
            "message": "Delete request processed",
            "deleted_ids": deleted_ids,
            "not_found_ids": not_found_ids,
            "tasks": serialise_tasks()
        }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)