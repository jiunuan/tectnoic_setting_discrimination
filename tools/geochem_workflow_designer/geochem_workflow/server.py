from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .catalog import NODE_TEMPLATES
from .executor import cancel_run, get_run, list_runs, start_run
from .presets import build_default_workflow
from .settings import DEFAULT_HOST, DEFAULT_PORT, STATIC_DIR, WORKFLOW_DIR, build_runtime_context, ensure_project_dirs


def _safe_workflow_path(name: str) -> Path:
    filename = Path(name).name
    if not filename.endswith(".json"):
        filename += ".json"
    return WORKFLOW_DIR / filename


def ensure_default_workflow_file() -> None:
    ensure_project_dirs()
    path = _safe_workflow_path("modern_basalt_train_test_safe.json")
    if not path.exists():
        path.write_text(json.dumps(build_default_workflow(), indent=2, ensure_ascii=False), encoding="utf-8")


class WorkflowHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/catalog":
            return self._send_json({"catalog": NODE_TEMPLATES, "context": build_runtime_context()})
        if parsed.path == "/api/preset/default":
            return self._send_json({"workflow": build_default_workflow()})
        if parsed.path == "/api/workflows":
            ensure_default_workflow_file()
            items = sorted(path.name for path in WORKFLOW_DIR.glob("*.json"))
            return self._send_json({"workflows": items})
        if parsed.path.startswith("/api/workflows/"):
            name = unquote(parsed.path.split("/api/workflows/", 1)[1])
            path = _safe_workflow_path(name)
            if not path.exists():
                return self._send_json({"error": "Workflow not found."}, status=HTTPStatus.NOT_FOUND)
            workflow = json.loads(path.read_text(encoding="utf-8"))
            return self._send_json({"workflow": workflow})
        if parsed.path == "/api/runs":
            return self._send_json({"runs": list_runs()})
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.split("/api/runs/", 1)[1]
            run = get_run(run_id)
            if run is None:
                return self._send_json({"error": "Run not found."}, status=HTTPStatus.NOT_FOUND)
            return self._send_json({"run": run})
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/workflows/save":
            body = self._read_json()
            workflow = body.get("workflow", {})
            filename = body.get("filename", "workflow.json")
            path = _safe_workflow_path(filename)
            path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False), encoding="utf-8")
            return self._send_json({"saved": True, "path": str(path)})
        if parsed.path == "/api/runs":
            body = self._read_json()
            workflow = body.get("workflow", {})
            run_id = start_run(workflow)
            return self._send_json({"run_id": run_id}, status=HTTPStatus.ACCEPTED)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/cancel"):
            run_id = parsed.path.split("/api/runs/", 1)[1].rsplit("/cancel", 1)[0]
            cancelled = cancel_run(run_id)
            if not cancelled:
                return self._send_json({"error": "Run not found or not running."}, status=HTTPStatus.CONFLICT)
            return self._send_json({"cancelled": True}, status=HTTPStatus.ACCEPTED)
        return self._send_json({"error": "Unsupported endpoint."}, status=HTTPStatus.NOT_FOUND)


def serve_http(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    ensure_default_workflow_file()
    server = ThreadingHTTPServer((host, port), WorkflowHandler)
    print(f"Workflow API running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    parser = argparse.ArgumentParser(description="Geochemical workflow designer")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    serve_http(args.host, args.port)
