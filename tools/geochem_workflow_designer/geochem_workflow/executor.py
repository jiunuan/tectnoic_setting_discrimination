from __future__ import annotations

import copy
import subprocess
import threading
import traceback
import uuid
from collections import deque
from datetime import datetime

from .catalog import NODE_TEMPLATES_BY_TYPE
from .operations import OPERATIONS, WorkflowCancelled
from .settings import build_runtime_context


RUNS: dict[str, dict] = {}
RUN_CONTROLS: dict[str, dict] = {}
RUNS_LOCK = threading.Lock()
RUN_STATUS_SORT_ORDER = {
    "running": 0,
    "failed": 1,
    "cancelled": 2,
    "completed": 3,
}
COMPLETED_NODE_STATUSES = {"completed", "skipped"}


def _update_run(run_id: str, **kwargs) -> None:
    with RUNS_LOCK:
        RUNS[run_id].update(kwargs)


def _append_log(run_id: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    with RUNS_LOCK:
        RUNS[run_id]["logs"].append(f"[{timestamp}] {message}")


def _set_node_status(run_id: str, node_id: str, status: str, detail: str | None = None) -> None:
    with RUNS_LOCK:
        RUNS[run_id]["node_statuses"][node_id] = {"status": status, "detail": detail or ""}


def _get_cancel_event(run_id: str):
    with RUNS_LOCK:
        control = RUN_CONTROLS.get(run_id)
        return control.get("cancel_event") if control else None


def _register_process(run_id: str, process: subprocess.Popen) -> None:
    with RUNS_LOCK:
        control = RUN_CONTROLS.get(run_id)
        if control is not None:
            control["current_process"] = process


def _clear_process(run_id: str, process: subprocess.Popen | None = None) -> None:
    with RUNS_LOCK:
        control = RUN_CONTROLS.get(run_id)
        if control is None:
            return
        current = control.get("current_process")
        if process is None or current is process:
            control["current_process"] = None


def _mark_running_nodes_cancelled(run_id: str, detail: str) -> None:
    with RUNS_LOCK:
        statuses = RUNS.get(run_id, {}).get("node_statuses", {})
        for node_id, node_state in statuses.items():
            if node_state.get("status") == "running":
                statuses[node_id] = {"status": "cancelled", "detail": detail}


def _build_run_node_metadata(workflow: dict) -> tuple[dict[str, str], list[str]]:
    node_titles: dict[str, str] = {}
    runnable_node_ids: list[str] = []
    for node in workflow.get("nodes", []):
        node_id = node.get("id")
        if not node_id:
            continue
        node_titles[node_id] = node.get("title") or node.get("type") or node_id
        template = NODE_TEMPLATES_BY_TYPE.get(node.get("type"))
        if template is None or template.get("runnable", True):
            runnable_node_ids.append(node_id)
    return node_titles, runnable_node_ids


def _get_focus_node_id(run: dict) -> str:
    node_statuses = run.get("node_statuses", {})
    runnable_node_ids = run.get("runnable_node_ids") or list(run.get("node_titles", {}).keys())

    for preferred_status in ("running", "failed", "cancelled"):
        for node_id in runnable_node_ids:
            if node_statuses.get(node_id, {}).get("status") == preferred_status:
                return node_id

    for node_id in reversed(runnable_node_ids):
        if node_id in node_statuses:
            return node_id

    return runnable_node_ids[0] if runnable_node_ids else ""


def _summarize_run(run: dict) -> dict:
    node_statuses = run.get("node_statuses", {})
    runnable_node_ids = run.get("runnable_node_ids") or list(run.get("node_titles", {}).keys())
    focus_node_id = _get_focus_node_id(run)
    total_nodes = len(runnable_node_ids)
    completed_nodes = sum(
        1
        for node_id in runnable_node_ids
        if node_statuses.get(node_id, {}).get("status") in COMPLETED_NODE_STATUSES
    )
    progress_percent = int(round((completed_nodes / total_nodes) * 100)) if total_nodes else 0
    last_log = run.get("logs", [])[-1] if run.get("logs") else ""

    return {
        "id": run.get("id", ""),
        "workflow_name": run.get("workflow_name", "unnamed"),
        "status": run.get("status", "idle"),
        "created_at": run.get("created_at", ""),
        "finished_at": run.get("finished_at", ""),
        "cancel_requested": run.get("cancel_requested", False),
        "error": run.get("error", ""),
        "log_count": len(run.get("logs", [])),
        "last_log": last_log,
        "current_node_id": focus_node_id,
        "current_node_title": run.get("node_titles", {}).get(focus_node_id, ""),
        "current_node_status": node_statuses.get(focus_node_id, {}).get("status", ""),
        "completed_nodes": completed_nodes,
        "total_nodes": total_nodes,
        "progress_percent": progress_percent,
    }


def topological_sort(workflow: dict) -> list[dict]:
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    node_map = {node["id"]: node for node in nodes}
    indegree = {node["id"]: 0 for node in nodes}
    adjacency = {node["id"]: [] for node in nodes}

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source not in node_map or target not in node_map:
            continue
        adjacency[source].append(target)
        indegree[target] += 1

    queue = deque([node_id for node_id, degree in indegree.items() if degree == 0])
    ordered = []
    while queue:
        node_id = queue.popleft()
        ordered.append(node_map[node_id])
        for neighbor in adjacency[node_id]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(nodes):
        raise RuntimeError("Workflow contains a cycle. Remove circular connections before running.")

    return ordered


def _execute(workflow: dict, run_id: str) -> None:
    context = build_runtime_context()
    context["_run_control"] = {
        "cancel_event": _get_cancel_event(run_id),
        "register_process": lambda process: _register_process(run_id, process),
        "clear_process": lambda process=None: _clear_process(run_id, process),
    }
    ordered_nodes = topological_sort(workflow)
    _append_log(run_id, f"Workflow: {workflow.get('name', 'unnamed')}")
    _append_log(run_id, f"Resolved runtime context: {context}")

    try:
        for node in ordered_nodes:
            cancel_event = _get_cancel_event(run_id)
            if cancel_event is not None and cancel_event.is_set():
                raise WorkflowCancelled("Workflow cancelled by user.")

            node_id = node["id"]
            node_type = node["type"]
            template = NODE_TEMPLATES_BY_TYPE.get(node_type)
            if template is None:
                raise KeyError(f"Unknown node type: {node_type}")

            if not template.get("runnable", True):
                _set_node_status(run_id, node_id, "skipped", "visual-only node")
                _append_log(run_id, f"{node['title']}: skipped (visual-only node)")
                continue

            _set_node_status(run_id, node_id, "running")
            _append_log(run_id, f"{node['title']}: running")
            op = OPERATIONS[node_type]
            try:
                result = op(node, context, lambda message: _append_log(run_id, f"{node['title']}: {message}"))
            except WorkflowCancelled:
                _set_node_status(run_id, node_id, "cancelled", "Workflow cancelled by user.")
                raise
            except Exception as exc:
                _set_node_status(run_id, node_id, "failed", str(exc))
                raise
            final_status = result.get("status", "completed")
            _set_node_status(run_id, node_id, final_status, result.get("detail"))
            _append_log(run_id, f"{node['title']}: {final_status}")

        _update_run(
            run_id,
            status="completed",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            cancel_requested=False,
        )
        _append_log(run_id, "Workflow finished successfully.")
    except WorkflowCancelled as exc:
        _mark_running_nodes_cancelled(run_id, str(exc))
        _update_run(
            run_id,
            status="cancelled",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            error="",
            traceback="",
            cancel_requested=True,
        )
        _append_log(run_id, f"Workflow cancelled: {exc}")
    except Exception as exc:
        _update_run(
            run_id,
            status="failed",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            error=str(exc),
            traceback=traceback.format_exc(),
            cancel_requested=False,
        )
        _append_log(run_id, f"Workflow failed: {exc}")
        _append_log(run_id, traceback.format_exc())
    finally:
        _clear_process(run_id)
        with RUNS_LOCK:
            RUN_CONTROLS.pop(run_id, None)


def start_run(workflow: dict) -> str:
    run_id = uuid.uuid4().hex[:10]
    node_titles, runnable_node_ids = _build_run_node_metadata(workflow)
    initial = {
        "id": run_id,
        "status": "running",
        "error": "",
        "traceback": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": "",
        "logs": [],
        "node_statuses": {},
        "workflow_name": workflow.get("name", "unnamed"),
        "cancel_requested": False,
        "node_titles": node_titles,
        "runnable_node_ids": runnable_node_ids,
    }
    with RUNS_LOCK:
        RUNS[run_id] = initial
        RUN_CONTROLS[run_id] = {"cancel_event": threading.Event(), "current_process": None}

    worker = threading.Thread(target=_execute, args=(copy.deepcopy(workflow), run_id), daemon=True)
    worker.start()
    return run_id


def cancel_run(run_id: str) -> bool:
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        control = RUN_CONTROLS.get(run_id)
        if run is None or control is None:
            return False
        if run.get("status") != "running":
            return False
        run["cancel_requested"] = True
        control["cancel_event"].set()

    _append_log(run_id, "Cancellation requested by user.")
    return True


def get_run(run_id: str) -> dict | None:
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        return copy.deepcopy(run) if run else None


def list_runs() -> list[dict]:
    with RUNS_LOCK:
        runs = [copy.deepcopy(run) for run in RUNS.values()]

    runs.sort(key=lambda run: run.get("created_at", ""), reverse=True)
    runs.sort(key=lambda run: RUN_STATUS_SORT_ORDER.get(run.get("status", "idle"), 99))
    return [_summarize_run(run) for run in runs]
