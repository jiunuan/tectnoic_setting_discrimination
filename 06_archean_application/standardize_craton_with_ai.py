"""
使用OpenAI兼容大模型统一太古代预测表中的克拉通名称。

特点：
1. 以“数据来源 + 当前克拉通名”为地质单元调用AI，不逐样品重复请求；
2. 汇总SOURCE_LOCATION、REFERENCE、ROCK_NAME和年龄文本作为判定证据；
3. 保留Craton_original，将AI标准名称写回Craton；
4. 直接把进度写入最终输出CSV，不生成报告或单独缓存文件；
5. 支持429指数退避、并发请求和断点续跑。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
import requests

try:
    import yaml
except ImportError:
    yaml = None


# === 统一路径配置：所有数据路径来自 config/paths.py ===
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
from config.paths import ARCHEAN_POOL_DIR, ARCHEAN_S3_CSV

DEFAULT_INPUT_CSV = str(ARCHEAN_POOL_DIR / "expanded_archean_predictions.csv")
DEFAULT_OUTPUT_CSV = str(ARCHEAN_POOL_DIR / "expanded_archean_basalt.csv")
LEGACY_AI_OUTPUT_CSV = str(ARCHEAN_POOL_DIR / "expanded_archean_predictions_craton_ai.csv")
LIU_REFERENCE_CSV = str(ARCHEAN_S3_CSV)
# 中文注释：LLM 接入配置（OPENAI_MODEL / OPENAI_BASE_URL / OPENAI_API_KEY）。
# 优先级：命令行参数 > 环境变量 > .env 文件 > YAML 配置。文件不存在时自动跳过。
DEFAULT_CONFIG_YAML = str(_PROJECT_ROOT / "config" / "llm_config.yaml")
FALLBACK_CONFIG_YAML = str(_PROJECT_ROOT / "config" / "llm_config.example.yaml")
DEFAULT_ENV_FILE = str(_PROJECT_ROOT / ".env")

ALLOWED_STATUS = {"CRATON", "PROVINCE_OR_SHIELD", "NOT_A_CRATON", "UNKNOWN"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}

# 中文注释：Windows终端默认GBK时，确保北欧地名等Unicode字符可以正常输出。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SYSTEM_PROMPT = """
你是一位熟悉前寒武纪地质、全球克拉通、地盾、构造省和绿岩带的资深地质学家。

任务：根据一个太古代样品组的现有Craton值、GeoROC层级LOCATION、文献题名或引用、
岩石名称、年龄文本和数据来源，判断这些样品所属的标准克拉通名称。

严格要求：
1. 输出全球地质文献中常用的英文标准名称，统一使用Title Case，例如：
   Yilgarn Craton、Pilbara Craton、Superior Craton、Kaapvaal Craton、
   Dharwar Craton、North China Craton、North Atlantic Craton。
2. Abitibi、Wawa、Uchi、Pontiac等Superior内部单元应归并为Superior Craton；
   Barberton归入Kaapvaal Craton；Isua和西南格陵兰太古代单元通常归入North Atlantic Craton。
3. 如果输入名称是同一克拉通的方位分区，例如Eastern Dharwar或Western Dharwar，
   标准名称仍应归并为Dharwar Craton。
4. 不要把绿岩带、超地体、岩群、火成岩省、现代火山弧或国家名称伪装成克拉通。
   若能可靠判断其母克拉通，输出母克拉通；否则status使用PROVINCE_OR_SHIELD、
   NOT_A_CRATON或UNKNOWN。
5. SOURCE_LOCATION通常是从大尺度到小尺度的“/”分层路径。优先使用其中明确的CRATON，
   但要识别同义词和下属构造单元。
6. 只依据提供的信息和可靠的地质常识。证据不足时必须降低置信度，不得编造。
7. canonical_craton必须保持简洁，不附加国家、方位说明、绿岩带或解释文字。
8. 只输出一个JSON对象，不要输出Markdown。

JSON字段：
{
  "canonical_craton": "标准英文名称；无法确定时为Unresolved",
  "status": "CRATON|PROVINCE_OR_SHIELD|NOT_A_CRATON|UNKNOWN",
  "confidence": "high|medium|low",
  "reason": "不超过80字的中文依据"
}
""".strip()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用AI统一扩展太古代预测表的克拉通名称")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--config", default=DEFAULT_CONFIG_YAML)
    parser.add_argument("--env", default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--max-units", type=int, default=0, help="0表示处理全部地质单元")
    parser.add_argument("--dry-run", action="store_true", help="只检查分组，不调用AI")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="直接更新输入CSV；原值仍保存在Craton_original",
    )
    return parser.parse_args()


def load_env_file(file_path: str) -> dict[str, str]:
    """读取简单KEY=VALUE格式的.env文件。"""
    values: dict[str, str] = {}
    if not file_path or not os.path.exists(file_path):
        return values
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_yaml_file(file_path: str) -> dict[str, Any]:
    """读取YAML配置；不存在config.yaml时回退到参考项目的示例配置。"""
    selected_path = file_path
    if not os.path.exists(selected_path) and os.path.exists(FALLBACK_CONFIG_YAML):
        selected_path = FALLBACK_CONFIG_YAML
    if not os.path.exists(selected_path) or yaml is None:
        return {}
    with open(selected_path, "r", encoding="utf-8", errors="replace") as file:
        data = yaml.safe_load(file)
    return data if isinstance(data, dict) else {}


def first_value(*values: Any) -> str | None:
    """返回第一个非空配置值。"""
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def first_int(*values: Any, default: int) -> int:
    """返回第一个有效整数配置。"""
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def load_api_config(args: argparse.Namespace) -> dict[str, Any]:
    """按命令行、环境变量、.env、YAML顺序加载API配置。"""
    env_file = load_env_file(args.env)
    yaml_file = load_yaml_file(args.config)
    yaml_llm = yaml_file.get("llm", {}) if isinstance(yaml_file.get("llm"), dict) else {}

    def yaml_value(*keys: str) -> Any:
        for key in keys:
            if yaml_file.get(key) not in (None, ""):
                return yaml_file[key]
            if yaml_llm.get(key) not in (None, ""):
                return yaml_llm[key]
        return None

    model = first_value(
        args.model,
        os.environ.get("OPENAI_MODEL"),
        env_file.get("OPENAI_MODEL"),
        yaml_value("openai-model", "model"),
    )
    base_url = first_value(
        args.base_url,
        os.environ.get("OPENAI_BASE_URL"),
        env_file.get("OPENAI_BASE_URL"),
        yaml_value("openai-base-url", "base_url"),
    )
    api_key = first_value(
        args.api_key,
        os.environ.get("OPENAI_API_KEY"),
        env_file.get("OPENAI_API_KEY"),
        yaml_value("openai-api-key", "api_key"),
    )
    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "timeout": first_int(
            args.timeout,
            yaml_value("llm-timeout", "timeout"),
            default=120,
        ),
        "max_tokens": first_int(
            args.max_tokens,
            yaml_value("llm-max-tokens", "max_tokens"),
            default=4000,
        ),
        "max_retries": first_int(
            args.max_retries,
            yaml_value("llm-max-retries", "max_retries"),
            default=8,
        ),
        "concurrency": first_int(
            args.concurrency,
            yaml_value("llm-concurrency", "concurrency"),
            default=6,
        ),
    }


def clean_text(value: Any) -> str:
    """压缩文本空白并移除无意义nan字符串。"""
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() == "nan" else text


def top_values(series: pd.Series, limit: int, text_limit: int = 500) -> list[str]:
    """提取频率最高的非空证据文本。"""
    cleaned = series.map(clean_text)
    cleaned = cleaned[cleaned.ne("")]
    values = cleaned.value_counts().head(limit).index.tolist()
    return [value[:text_limit] for value in values]


def build_units(data: pd.DataFrame) -> pd.DataFrame:
    """按来源和当前克拉通名称聚合AI判定单元。"""
    required = ["SOURCE_DATASET", "Craton"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"输入CSV缺少字段: {missing}")

    working = data.copy()
    working["_craton_group"] = working["Craton"].map(clean_text)
    working.loc[working["_craton_group"].eq(""), "_craton_group"] = "Unresolved"

    rows = []
    for (source, craton), group in working.groupby(
        ["SOURCE_DATASET", "_craton_group"],
        dropna=False,
        sort=True,
    ):
        rows.append(
            {
                "unit_id": f"{clean_text(source)}|{craton}",
                "SOURCE_DATASET": clean_text(source),
                "current_craton": craton,
                "n_samples": len(group),
                "source_locations": top_values(
                    group["SOURCE_LOCATION"]
                    if "SOURCE_LOCATION" in group.columns
                    else pd.Series(dtype=object),
                    12,
                    700,
                ),
                "references": top_values(
                    group["REFERENCE"]
                    if "REFERENCE" in group.columns
                    else pd.Series(dtype=object),
                    6,
                    500,
                ),
                "rock_names": top_values(
                    group["ROCK_NAME"]
                    if "ROCK_NAME" in group.columns
                    else pd.Series(dtype=object),
                    8,
                    120,
                ),
                "age_texts": top_values(
                    group["SOURCE_AGE_TEXT"]
                    if "SOURCE_AGE_TEXT" in group.columns
                    else pd.Series(dtype=object),
                    5,
                    200,
                ),
                "original_tectonic_labels": top_values(
                    group["SOURCE_ORIGINAL_TECTONIC_LABEL"]
                    if "SOURCE_ORIGINAL_TECTONIC_LABEL" in group.columns
                    else pd.Series(dtype=object),
                    8,
                    160,
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """从模型响应中逐个提取平衡的JSON对象。"""
    if not text:
        return None

    # 中文注释：推理模型可能在最终JSON前后输出解释，不能直接截取首尾花括号。
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (
            "canonical_craton" in parsed or "status" in parsed
        ):
            return parsed
    return None


def sanitize_result(result: dict[str, Any]) -> dict[str, str]:
    """约束AI返回字段，防止异常值污染CSV。"""
    canonical = clean_text(result.get("canonical_craton", ""))
    status = clean_text(result.get("status", "")).upper()
    confidence = clean_text(result.get("confidence", "")).lower()
    reason = clean_text(result.get("reason", ""))[:160]

    if status not in ALLOWED_STATUS:
        status = "UNKNOWN"
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "low"
    if not canonical:
        canonical = "Unresolved"
    if status in {"NOT_A_CRATON", "UNKNOWN"} and canonical.lower() in {
        "not a craton",
        "unknown",
        "unresolved",
    }:
        canonical = "Unresolved"
    return {
        "canonical_craton": canonical,
        "status": status,
        "confidence": confidence,
        "reason": reason,
    }


def call_ai(unit: dict[str, Any], config: dict[str, Any]) -> dict[str, str] | None:
    """调用OpenAI兼容chat/completions接口并处理429退避。"""
    url = config["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "SOURCE_DATASET": unit["SOURCE_DATASET"],
        "current_craton": unit["current_craton"],
        "n_samples": unit["n_samples"],
        "source_locations": unit["source_locations"],
        "references": unit["references"],
        "rock_names": unit["rock_names"],
        "age_texts": unit["age_texts"],
        "original_tectonic_labels": unit["original_tectonic_labels"],
    }
    body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请判断以下样品组所属的标准克拉通：\n"
                + json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
        "temperature": 0.0,
        "max_tokens": config["max_tokens"],
    }

    for attempt in range(config["max_retries"] + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=config["timeout"],
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "", 1).isdigit()
                    else min(90.0, 2.0 ** attempt + attempt)
                )
                print(
                    f"[429] {unit['unit_id']}，等待{wait_seconds:.1f}秒后重试 "
                    f"({attempt + 1}/{config['max_retries']})"
                )
                time.sleep(wait_seconds)
                continue
            if response.status_code != 200:
                print(
                    f"[失败] {unit['unit_id']} HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return None

            response_data = response.json()
            choice = response_data["choices"][0]
            message = choice.get("message", {}) or {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            parsed = parse_json_object(content)
            if parsed is None:
                parsed = parse_json_object(reasoning)
            if parsed is None:
                finish_reason = choice.get("finish_reason", "")
                usage = response_data.get("usage", {})
                reasoning_tokens = (
                    usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                )
                print(
                    f"[解析重试] {unit['unit_id']} finish={finish_reason}, "
                    f"reasoning_tokens={reasoning_tokens}, "
                    f"content={content[:100]!r}"
                )
                if attempt < config["max_retries"]:
                    time.sleep(min(10.0, 1.5 ** attempt))
                    continue
                print(f"[失败] {unit['unit_id']} 响应无法解析为JSON")
                return None
            return sanitize_result(parsed)
        except requests.RequestException as exception:
            if attempt >= config["max_retries"]:
                print(f"[失败] {unit['unit_id']} {type(exception).__name__}: {exception}")
                return None
            wait_seconds = min(60.0, 2.0 ** attempt)
            time.sleep(wait_seconds)
    return None


def existing_result_map(data: pd.DataFrame) -> dict[str, dict[str, str]]:
    """从已有输出CSV读取已完成结果，实现断点续跑。"""
    required = {
        "Craton_original",
        "Craton_AI",
        "Craton_AI_status",
        "Craton_AI_confidence",
        "Craton_AI_reason",
    }
    if not required.issubset(data.columns):
        return {}

    result: dict[str, dict[str, str]] = {}
    for (source, original), group in data.groupby(
        ["SOURCE_DATASET", "Craton_original"],
        dropna=False,
    ):
        first = group.iloc[0]
        canonical = clean_text(first["Craton_AI"])
        if not canonical:
            continue
        unit_id = f"{clean_text(source)}|{clean_text(original) or 'Unresolved'}"
        result[unit_id] = {
            "canonical_craton": canonical,
            "status": clean_text(first["Craton_AI_status"]),
            "confidence": clean_text(first["Craton_AI_confidence"]),
            "reason": clean_text(first["Craton_AI_reason"]),
        }
    return result


def apply_results(
    original_data: pd.DataFrame,
    result_map: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """把单元级AI结果回填到每条样品，并保留原克拉通列。"""
    output = original_data.copy()
    if "Craton_original" not in output.columns:
        output.insert(
            output.columns.get_loc("Craton"),
            "Craton_original",
            output["Craton"].to_numpy(),
        )

    ai_craton = []
    ai_status = []
    ai_confidence = []
    ai_reason = []
    final_craton = []

    for _, row in output.iterrows():
        original = clean_text(row["Craton_original"]) or "Unresolved"
        unit_id = f"{clean_text(row['SOURCE_DATASET'])}|{original}"
        result = result_map.get(unit_id)
        if result is None:
            ai_craton.append("")
            ai_status.append("")
            ai_confidence.append("")
            ai_reason.append("")
            final_craton.append(original)
            continue

        canonical = result["canonical_craton"]
        ai_craton.append(canonical)
        ai_status.append(result["status"])
        ai_confidence.append(result["confidence"])
        ai_reason.append(result["reason"])

        # 中文注释：只有中高置信且确认为克拉通时才自动替换原列。
        if (
            result["status"] == "CRATON"
            and result["confidence"] in {"high", "medium"}
            and canonical != "Unresolved"
        ):
            final_craton.append(canonical)
        else:
            final_craton.append(original)

    output["Craton"] = final_craton
    output["Craton_AI"] = ai_craton
    output["Craton_AI_status"] = ai_status
    output["Craton_AI_confidence"] = ai_confidence
    output["Craton_AI_reason"] = ai_reason
    return output


def save_output(data: pd.DataFrame, output_path: str) -> None:
    """原子保存最终CSV，避免中断时留下空文件。"""
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    # 中文注释：先写同目录临时文件，完整写入后再替换正式输出。
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            suffix=".csv",
            dir=output_directory or None,
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            data.to_csv(temporary_file, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def organize_like_liu(process_data: pd.DataFrame) -> pd.DataFrame:
    """按Liu共享文件的57列结构组织最终扩展太古代数据。"""
    # 中文注释：06_archean_application 目录不是Python包，显式加入完整目录后导入。
    _archean_dir = str(_PROJECT_ROOT / "06_archean_application")
    if _archean_dir not in sys.path:
        sys.path.insert(0, _archean_dir)
    from extended_archean_pool_analysis import (
        build_georoc_recovered_pool,
        build_liu_pool,
        combine_and_deduplicate,
    )

    liu_pool = build_liu_pool()
    georoc_pool, _ = build_georoc_recovered_pool()
    expanded_pool, _ = combine_and_deduplicate(liu_pool, georoc_pool)
    if len(expanded_pool) != len(process_data):
        raise RuntimeError(
            f"扩展池行数({len(expanded_pool)})与预测结果行数"
            f"({len(process_data)})不一致，不能按行回填。"
        )

    # 中文注释：样品顺序来自同一扩展池流程，先核对关键字段再回填结果。
    pool_ids = expanded_pool["SAMPLE_ID"].fillna("").astype(str).to_numpy()
    prediction_ids = process_data["SAMPLE_ID"].fillna("").astype(str).to_numpy()
    if not (pool_ids == prediction_ids).all():
        raise RuntimeError("扩展池与预测结果的SAMPLE_ID顺序不一致。")

    liu_columns = pd.read_csv(LIU_REFERENCE_CSV, nrows=0).columns.tolist()
    final_data = expanded_pool.reindex(columns=liu_columns).copy()
    final_data["Craton"] = process_data["Craton"].to_numpy()

    # 中文注释：Liu文件中的Arc_probability3与三类弧概率和概念一致。
    final_data["Arc_probability3"] = pd.to_numeric(
        process_data["P_arc"],
        errors="coerce",
    ).to_numpy()
    return final_data


def main() -> None:
    args = parse_args()
    output_path = args.input if args.in_place else args.output

    # 中文注释：兼容读取上一版带过程字段的结果，仅用于复用已完成判定。
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        output_header = pd.read_csv(output_path, nrows=0).columns
    else:
        output_header = []
    if "Craton_AI" in output_header:
        resume_path = output_path
    elif os.path.exists(LEGACY_AI_OUTPUT_CSV) and os.path.getsize(LEGACY_AI_OUTPUT_CSV) > 0:
        resume_path = LEGACY_AI_OUTPUT_CSV
    else:
        resume_path = args.input
    data = pd.read_csv(resume_path, low_memory=False)
    units = build_units(
        data.assign(
            Craton=(
                data["Craton_original"]
                if "Craton_original" in data.columns
                else data["Craton"]
            )
        )
    )
    completed = existing_result_map(data)
    pending = units[~units["unit_id"].isin(completed)].copy()
    if args.max_units > 0:
        pending = pending.head(args.max_units)

    print(f"输入样品数: {len(data)}")
    print(f"唯一判定单元: {len(units)}")
    print(f"已完成单元: {len(completed)}")
    print(f"本次待调用单元: {len(pending)}")

    if args.dry_run:
        print(
            pending[
                ["unit_id", "n_samples", "current_craton", "source_locations"]
            ].to_string(index=False)
        )
        return

    config = load_api_config(args)
    masked_key = (
        f"{config['api_key'][:4]}...{config['api_key'][-4:]}"
        if config.get("api_key") and len(config["api_key"]) >= 8
        else "未配置"
    )
    print(
        "API配置: "
        f"model={config['model']}, base_url={config['base_url']}, "
        f"max_tokens={config['max_tokens']}, concurrency={config['concurrency']}, "
        f"api_key={masked_key}"
    )
    missing_config = [
        key for key in ["model", "base_url", "api_key"] if not config.get(key)
    ]
    if missing_config:
        raise RuntimeError(
            "缺少API配置: "
            + ", ".join(missing_config)
            + "。请配置OPENAI_MODEL、OPENAI_BASE_URL、OPENAI_API_KEY。"
        )

    if len(pending):
        workers = max(1, config["concurrency"])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(call_ai, unit.to_dict(), config): unit.to_dict()
                for _, unit in pending.iterrows()
            }
            done = 0
            for future in as_completed(futures):
                unit = futures[future]
                result = future.result()
                done += 1
                if result is not None:
                    completed[unit["unit_id"]] = result
                    print(
                        f"[{done}/{len(pending)}] {unit['current_craton']} -> "
                        f"{result['canonical_craton']} "
                        f"({result['status']}, {result['confidence']})"
                    )
                else:
                    print(f"[{done}/{len(pending)}] {unit['current_craton']} -> 调用失败")

                # 中文注释：调用期间保存过程结果，全部完成后转换为Liu字段结构。
                current_output = apply_results(data, completed)
                save_output(current_output, LEGACY_AI_OUTPUT_CSV)

    process_output = apply_results(data, completed)
    final_output = organize_like_liu(process_output)
    save_output(final_output, output_path)
    standardized = final_output["Craton"].nunique(dropna=False)
    print(f"输出文件: {output_path}")
    print(f"标准化后Craton唯一值: {standardized}")
    print(f"最终字段数: {len(final_output.columns)}")


if __name__ == "__main__":
    main()
