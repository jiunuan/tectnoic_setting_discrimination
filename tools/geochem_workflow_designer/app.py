from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from geochem_workflow.server import serve_http
from geochem_workflow.settings import DEFAULT_HOST, DEFAULT_PORT, FRONTEND_DEV_HOST, FRONTEND_DEV_PORT


PROJECT_DIR = Path(__file__).resolve().parent


def _kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def _resolve_npm_command() -> str:
    if os.name == "nt":
        return shutil.which("npm.cmd") or shutil.which("npm") or "npm.cmd"
    return shutil.which("npm") or "npm"


def start_frontend_dev(frontend_host: str, frontend_port: int) -> subprocess.Popen:
    npm_cmd = _resolve_npm_command()
    command = [npm_cmd, "run", "dev", "--", "--host", frontend_host, "--port", str(frontend_port)]
    print(f"Starting Vue dev server: {' '.join(command)}")
    return subprocess.Popen(command, cwd=str(PROJECT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the geochem workflow backend and optional Vue dev frontend.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Backend host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Backend port")
    parser.add_argument("--frontend-host", default=FRONTEND_DEV_HOST, help="Frontend dev server host")
    parser.add_argument("--frontend-port", type=int, default=FRONTEND_DEV_PORT, help="Frontend dev server port")
    parser.add_argument("--backend-only", action="store_true", help="Run only the Python backend API")
    args = parser.parse_args()

    frontend_process: subprocess.Popen | None = None
    try:
        if not args.backend_only:
            frontend_process = start_frontend_dev(args.frontend_host, args.frontend_port)
            print(f"Vue app running at http://{args.frontend_host}:{args.frontend_port}")
            print(f"Python API running at http://{args.host}:{args.port}")
        else:
            print(f"Python API running at http://{args.host}:{args.port}")
        serve_http(args.host, args.port)
    finally:
        if frontend_process is not None:
            _kill_process_tree(frontend_process)


if __name__ == "__main__":
    main()
