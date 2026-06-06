from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = PROJECT_DIR / "geochem_workflow"
LEGACY_STATIC_DIR = PACKAGE_DIR / "static"
FRONTEND_DIST_DIR = PROJECT_DIR / "frontend_dist"
STATIC_DIR = FRONTEND_DIST_DIR if FRONTEND_DIST_DIR.exists() else LEGACY_STATIC_DIR
WORKFLOW_DIR = PROJECT_DIR / "workflows"
RUN_DIR = PROJECT_DIR / "runs"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

DEFAULT_HOST = "127.0.0.1"
# 后端默认端口避开本机代理常用的 8765 端口。
DEFAULT_PORT = 8766
FRONTEND_DEV_HOST = "127.0.0.1"
FRONTEND_DEV_PORT = 5173

# 本设计器位于 <项目根>/tools/geochem_workflow_designer/；
# repo_root 指向主项目根目录（basalt_tectonic_discrimination），各脚本路径据此推导。
REPO_ROOT = PROJECT_DIR.parent.parent


def ensure_project_dirs() -> None:
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def build_runtime_context() -> dict[str, str]:
    ensure_project_dirs()
    return {
        "python_exe": sys.executable,
        "repo_root": str(REPO_ROOT),
        "workflow_root": str(PROJECT_DIR),
        "artifact_root": str(ARTIFACT_DIR),
        "today": datetime.now().strftime("%Y-%m-%d"),
    }
