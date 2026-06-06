from __future__ import annotations

import json
import os
import pickle
import subprocess
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


LogFn = Callable[[str], None]


class WorkflowCancelled(RuntimeError):
    pass


def _get_run_control(context: dict[str, str]) -> dict:
    return context.get("_run_control", {}) if isinstance(context, dict) else {}


def _raise_if_cancelled(context: dict[str, str]) -> None:
    cancel_event = _get_run_control(context).get("cancel_event")
    if cancel_event is not None and cancel_event.is_set():
        raise WorkflowCancelled("Workflow cancelled by user.")


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


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def resolve_text(value: str, context: dict[str, str]) -> str:
    return str(value).format_map(SafeFormatDict(context))


def resolve_path(value: str, context: dict[str, str]) -> Path:
    return Path(resolve_text(value, context)).expanduser()


def parse_lines(value: str) -> list[str]:
    if not value:
        return []
    normalized = value.replace(",", "\n").replace(";", "\n")
    items = [item.strip() for item in normalized.splitlines()]
    return [item for item in items if item]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_with_fallback(path: Path, encodings: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    tried_encodings = encodings or ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1"]
    last_error: Exception | None = None
    for encoding in tried_encodings:
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read CSV: {path}") from last_error


def run_note(node: dict, context: dict[str, str], log: LogFn) -> dict:
    log("Note node skipped.")
    return {"status": "skipped"}


def _stream_process_output(pipe, log: LogFn, prefix: str = "") -> None:
    if pipe is None:
        return
    try:
        for line in iter(pipe.readline, ""):
            if line == "":
                break
            message = line.rstrip("\r\n")
            if prefix:
                log(f"{prefix}{message}")
            else:
                log(message)
    finally:
        pipe.close()


def run_command_task(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    command = resolve_text(params.get("command", ""), context).strip()
    if not command:
        log("Command is empty. Skipped.")
        return {"status": "skipped"}

    _raise_if_cancelled(context)
    control = _get_run_control(context)
    register_process = control.get("register_process")
    clear_process = control.get("clear_process")
    cancel_event = control.get("cancel_event")

    workdir = Path(resolve_text(params.get("workdir", "."), context))
    workdir.mkdir(parents=True, exist_ok=True)
    log(f"Working directory: {workdir}")
    log(f"Command: {command}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")

    process = subprocess.Popen(
        command,
        cwd=str(workdir),
        shell=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if callable(register_process):
        register_process(process)

    stdout_thread = threading.Thread(target=_stream_process_output, args=(process.stdout, log), daemon=True)
    stderr_thread = threading.Thread(target=_stream_process_output, args=(process.stderr, log, "[stderr] "), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    cancelled = False
    try:
        while True:
            try:
                return_code = process.wait(timeout=0.3)
                break
            except subprocess.TimeoutExpired:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    log("Cancellation requested. Stopping current process...")
                    _kill_process_tree(process)
                    return_code = process.wait()
                    break
    finally:
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        if callable(clear_process):
            clear_process(process)

    if cancelled:
        raise WorkflowCancelled("Workflow cancelled by user.")
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}.")

    return {"status": "completed", "return_code": return_code}


def run_rebuild_refined_basalt_labels(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    repo_root = resolve_text("{repo_root}", context)
    script_path = Path(repo_root) / "data_preprocess" / "convergent_margin_process" / "rebuild_refined_basalt_from_revised_labels.py"
    command = (
        f'"{resolve_text("{python_exe}", context)}" '
        f'"{script_path}" '
        f'--sea "{resolve_text(params.get("sea_path", ""), context)}" '
        f'--land "{resolve_text(params.get("land_path", ""), context)}" '
        f'--merged-raw "{resolve_text(params.get("merged_raw_path", ""), context)}" '
        f'--convergent-output "{resolve_text(params.get("convergent_output_path", ""), context)}" '
        f'--basalt-output "{resolve_text(params.get("basalt_output_path", ""), context)}"'
    )
    command_node = {
        "params": {
            "command": command,
            "workdir": params.get("workdir", "{repo_root}"),
        }
    }
    return run_command_task(command_node, context, log)


def run_cnn_bilstm_train(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    repo_root = resolve_text("{repo_root}", context)
    script_path = Path(repo_root) / "model_train" / "CNN-BiLSTM" / "cnn_bilstm_presplit.py"
    command = (
        f'"{resolve_text("{python_exe}", context)}" '
        f'"{script_path}" '
        f'--train-file "{resolve_text(params.get("train_file", ""), context)}" '
        f'--test-file "{resolve_text(params.get("test_file", ""), context)}" '
        f'--output-dir "{resolve_text(params.get("output_dir", ""), context)}" '
        f'--seed {int(params.get("seed", 42))} '
        f'--batch-size {int(params.get("batch_size", 64))} '
        f'--epochs {int(params.get("epochs", 200))} '
        f'--lr {float(params.get("lr", 0.0001))} '
        f'--weight-decay {float(params.get("weight_decay", 0.03))} '
        f'--scheduler-patience {int(params.get("scheduler_patience", 15))} '
        f'--early-stopping-patience {int(params.get("early_stopping_patience", 40))} '
        f'--mixup-alpha {float(params.get("mixup_alpha", 0.4))} '
        f'--label-smoothing {float(params.get("label_smoothing", 0.2))} '
        f'--file-stem "{resolve_text(params.get("file_stem", "basalt"), context)}" '
        f'--model-name "{resolve_text(params.get("model_name", "cnn_bilstm_best.pth"), context)}" '
        f'--results-name "{resolve_text(params.get("results_name", "CNN_BiLSTM_results.csv"), context)}"'
    )
    command_node = {
        "params": {
            "command": command,
            "workdir": params.get("workdir", "{repo_root}"),
        }
    }
    return run_command_task(command_node, context, log)


def run_extract_georoc_filter(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    repo_root = resolve_text("{repo_root}", context)
    script_path = Path(repo_root) / "data_preprocess" / "extract" / "extract_georoc.py"
    command = (
        f'"{resolve_text("{python_exe}", context)}" '
        f'"{script_path}" '
        f'--file-path "{resolve_text(params.get("file_path", ""), context)}" '
        f'--output-path "{resolve_text(params.get("output_path", ""), context)}"'
    )
    command_node = {
        "params": {
            "command": command,
            "workdir": params.get("workdir", "{repo_root}"),
        }
    }
    return run_command_task(command_node, context, log)


def run_merge_georoc_petdb(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    georoc_input_path = resolve_path(params["georoc_input_path"], context)
    petdb_input_path = resolve_path(params["petdb_input_path"], context)
    output_path = resolve_path(params["output_path"], context)

    inputs = [
        ("GEOROC", georoc_input_path),
        ("PetDB", petdb_input_path),
    ]

    frames = []
    for label, path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"{label} input not found: {path}")
        frame, encoding = read_csv_with_fallback(path)
        log(f"{label} loaded: {path} ({len(frame)} rows, encoding={encoding})")
        frames.append(frame)

    combined_df = pd.concat(frames, ignore_index=True, sort=False)
    ensure_parent(output_path)
    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    log(f"Combined output: {output_path} ({len(combined_df)} rows)")
    return {
        "status": "completed",
        "output_path": str(output_path),
        "rows": int(len(combined_df)),
    }


def run_stratified_split(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    input_path = resolve_path(params["input_path"], context)
    label_column = params["label_column"]
    test_size = float(params.get("test_size", 0.2))
    random_state = int(params.get("random_state", 42))
    output_dir = resolve_path(params["output_dir"], context)
    output_prefix = params.get("output_prefix", "dataset")

    df = pd.read_csv(input_path, low_memory=False)
    if label_column not in df.columns:
        raise KeyError(f"Label column not found: {label_column}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    train_parts = []
    test_parts = []
    warnings = []
    for label, group in df.groupby(label_column, sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        group_size = len(indices)
        if group_size <= 1:
            warnings.append(f"Label '{label}' has only {group_size} row; kept in train.")
            train_parts.append(df.loc[indices])
            continue

        test_count = int(round(group_size * test_size))
        test_count = max(1, min(group_size - 1, test_count))
        test_index = indices[:test_count]
        train_index = indices[test_count:]
        train_parts.append(df.loc[train_index])
        test_parts.append(df.loc[test_index])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_path = output_dir / f"{output_prefix}_train.csv"
    test_path = output_dir / f"{output_prefix}_test.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    log(f"Train saved: {train_path} ({len(train_df)} rows)")
    log(f"Test saved:  {test_path} ({len(test_df)} rows)")
    for item in warnings:
        log(f"[warn] {item}")

    return {
        "status": "completed",
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }


def run_iqr_filter(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    input_path = resolve_path(params["input_path"], context)
    output_path = resolve_path(params["output_path"], context)
    feature_columns = parse_lines(params.get("feature_columns", ""))
    threshold = float(params.get("threshold", 6))

    df = pd.read_csv(input_path, low_memory=False)
    mask = pd.Series(False, index=df.index)
    for column in feature_columns:
        if column not in df.columns:
            log(f"[warn] Missing column skipped: {column}")
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        non_null = values.dropna()
        if non_null.empty:
            continue
        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        column_mask = (values < lower) | (values > upper)
        mask = mask | column_mask.fillna(False)

    filtered = df.loc[~mask].reset_index(drop=True)
    ensure_parent(output_path)
    filtered.to_csv(output_path, index=False, encoding="utf-8-sig")
    log(f"Rows removed: {int(mask.sum())}")
    log(f"Filtered output: {output_path} ({len(filtered)} rows)")
    return {"status": "completed", "rows_removed": int(mask.sum()), "output_path": str(output_path)}


def _clean_for_rf(df: pd.DataFrame) -> pd.DataFrame:
    capped = df.replace([np.inf, -np.inf], np.finfo(np.float32).max)
    return capped.fillna(0)


def _missforest_fit_transform(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    n_estimators: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as exc:
        raise RuntimeError(
            "MissForest node needs scipy + scikit-learn in the active Python environment."
        ) from exc

    train_numeric = train_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    test_numeric = test_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    train_clean = _clean_for_rf(train_numeric)
    test_clean = _clean_for_rf(test_numeric)

    models = {}
    for column in feature_columns:
        other_columns = [item for item in feature_columns if item != column]
        mask = train_numeric[column].notna()
        if int(mask.sum()) == 0:
            continue
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(train_clean.loc[mask, other_columns], train_numeric.loc[mask, column])
        models[column] = model

    train_imputed = train_numeric.copy()
    test_imputed = test_numeric.copy()
    for column in feature_columns:
        if column not in models:
            continue
        other_columns = [item for item in feature_columns if item != column]
        train_missing = train_imputed[column].isna()
        test_missing = test_imputed[column].isna()
        if train_missing.any():
            train_imputed.loc[train_missing, column] = models[column].predict(
                train_clean.loc[train_missing, other_columns]
            )
        if test_missing.any():
            test_imputed.loc[test_missing, column] = models[column].predict(
                test_clean.loc[test_missing, other_columns]
            )

    train_output = train_df.copy()
    test_output = test_df.copy()
    train_output.loc[:, feature_columns] = train_imputed
    test_output.loc[:, feature_columns] = test_imputed
    bundle = {"feature_columns": feature_columns, "models": models, "random_state": random_state}
    return train_output, test_output, bundle


def run_missforest_train_test(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    feature_columns = parse_lines(params.get("feature_columns", ""))
    train_input_path = resolve_path(params["train_input_path"], context)
    test_input_path = resolve_path(params["test_input_path"], context)
    train_output_path = resolve_path(params["train_output_path"], context)
    test_output_path = resolve_path(params["test_output_path"], context)
    model_output_path = resolve_path(params["model_output_path"], context)
    n_estimators = int(params.get("n_estimators", 300))
    random_state = int(params.get("random_state", 42))

    train_df = pd.read_csv(train_input_path, low_memory=False)
    test_df = pd.read_csv(test_input_path, low_memory=False)
    missing_columns = [column for column in feature_columns if column not in train_df.columns or column not in test_df.columns]
    if missing_columns:
        raise KeyError(f"Missing feature columns: {missing_columns}")

    train_output, test_output, bundle = _missforest_fit_transform(
        train_df,
        test_df,
        feature_columns,
        n_estimators,
        random_state,
    )

    ensure_parent(train_output_path)
    ensure_parent(test_output_path)
    ensure_parent(model_output_path)
    train_output.to_csv(train_output_path, index=False, encoding="utf-8-sig")
    test_output.to_csv(test_output_path, index=False, encoding="utf-8-sig")
    with model_output_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    log(f"Train imputed: {train_output_path}")
    log(f"Test imputed:  {test_output_path}")
    log(f"Model bundle:  {model_output_path}")
    return {
        "status": "completed",
        "train_output_path": str(train_output_path),
        "test_output_path": str(test_output_path),
        "model_output_path": str(model_output_path),
    }


def run_smote_balance(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    input_path = resolve_path(params["input_path"], context)
    output_path = resolve_path(params["output_path"], context)
    label_column = params["label_column"]
    target_count = int(params.get("target_count", 7000))
    random_state = int(params.get("random_state", 42))
    feature_columns = parse_lines(params.get("feature_columns", ""))

    try:
        from imblearn.over_sampling import SMOTE
    except Exception as exc:
        raise RuntimeError(
            "SMOTE node needs imbalanced-learn installed in the active Python environment."
        ) from exc

    df = pd.read_csv(input_path, low_memory=False)
    if label_column not in df.columns:
        raise KeyError(f"Label column not found: {label_column}")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    x = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = df[label_column].astype(str)

    counts = y.value_counts().to_dict()
    sampling_strategy = {
        label: target_count for label, count in counts.items() if count < target_count
    }
    if not sampling_strategy:
        log("All classes already satisfy the target count. Input copied to output.")
        ensure_parent(output_path)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return {"status": "completed", "output_path": str(output_path), "sampling_strategy": {}}

    min_class_size = min(counts.values())
    if min_class_size <= 1:
        raise RuntimeError("At least one class has <=1 sample; SMOTE cannot be applied safely.")

    k_neighbors = min(5, min_class_size - 1)
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    x_resampled, y_resampled = smote.fit_resample(x, y)

    result = pd.DataFrame(x_resampled, columns=feature_columns)
    result[label_column] = y_resampled
    other_columns = [column for column in df.columns if column not in feature_columns + [label_column]]
    for column in other_columns:
        result[column] = pd.NA

    ensure_parent(output_path)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    log(f"SMOTE output: {output_path} ({len(result)} rows)")
    return {"status": "completed", "output_path": str(output_path), "rows": int(len(result))}


def run_anhydrous_normalize(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    input_path = resolve_path(params["input_path"], context)
    output_path = resolve_path(params["output_path"], context)
    major_columns = parse_lines(params.get("major_columns", ""))

    df = pd.read_csv(input_path, low_memory=False)
    missing = [column for column in major_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing major columns: {missing}")

    result = df.copy()
    row_sum = result[major_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    zero_mask = row_sum.eq(0)
    if zero_mask.any():
        log(f"[warn] {int(zero_mask.sum())} rows have zero major-element sum.")
    for column in major_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce") / row_sum * 100

    ensure_parent(output_path)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    log(f"Anhydrous normalized output: {output_path}")
    return {"status": "completed", "output_path": str(output_path)}


def _fit_quantile_payload(df: pd.DataFrame, feature_columns: list[str], bins: int) -> dict:
    quantile_positions = np.linspace(0, 1, bins + 1)[1:-1]
    payload = {"bins": bins, "columns": {}}
    for column in feature_columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            payload["columns"][column] = []
            continue
        thresholds = np.quantile(values, quantile_positions, method="linear").tolist()
        payload["columns"][column] = thresholds
    return payload


def _apply_quantiles(df: pd.DataFrame, feature_columns: list[str], payload: dict) -> pd.DataFrame:
    bins = int(payload["bins"])
    result = df.copy()
    for column in feature_columns:
        if column not in result.columns:
            continue
        thresholds = np.array(payload["columns"].get(column, []), dtype=float)
        values = pd.to_numeric(result[column], errors="coerce")
        valid = values.notna()
        transformed = pd.Series(np.nan, index=result.index, dtype=float)
        if thresholds.size == 0:
            transformed.loc[valid] = bins
        else:
            transformed.loc[valid] = np.searchsorted(thresholds, values.loc[valid], side="left") + 1
        result[column] = transformed.clip(lower=1, upper=bins).astype("Int64")
    return result


def run_quantile_fit(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    input_path = resolve_path(params["input_path"], context)
    params_output_path = resolve_path(params["params_output_path"], context)
    transformed_output_path = resolve_path(params["transformed_output_path"], context)
    feature_columns = parse_lines(params.get("feature_columns", ""))
    bins = int(params.get("bins", 255))

    df = pd.read_csv(input_path, low_memory=False)
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    payload = _fit_quantile_payload(df, feature_columns, bins)
    transformed = _apply_quantiles(df, feature_columns, payload)

    ensure_parent(params_output_path)
    ensure_parent(transformed_output_path)
    params_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    transformed.to_csv(transformed_output_path, index=False, encoding="utf-8-sig")

    log(f"Quantile params: {params_output_path}")
    log(f"Train transformed: {transformed_output_path}")
    return {
        "status": "completed",
        "params_output_path": str(params_output_path),
        "transformed_output_path": str(transformed_output_path),
    }


def run_quantile_apply(node: dict, context: dict[str, str], log: LogFn) -> dict:
    params = node["params"]
    input_path = resolve_path(params["input_path"], context)
    params_input_path = resolve_path(params["params_input_path"], context)
    output_path = resolve_path(params["output_path"], context)
    feature_columns = parse_lines(params.get("feature_columns", ""))

    df = pd.read_csv(input_path, low_memory=False)
    payload = json.loads(params_input_path.read_text(encoding="utf-8"))
    transformed = _apply_quantiles(df, feature_columns, payload)

    ensure_parent(output_path)
    transformed.to_csv(output_path, index=False, encoding="utf-8-sig")
    log(f"Applied quantile bins: {output_path}")
    return {"status": "completed", "output_path": str(output_path)}


OPERATIONS = {
    "note": run_note,
    "command_task": run_command_task,
    "rebuild_refined_basalt_labels": run_rebuild_refined_basalt_labels,
    "cnn_bilstm_train": run_cnn_bilstm_train,
    "extract_georoc_filter": run_extract_georoc_filter,
    "merge_georoc_petdb": run_merge_georoc_petdb,
    "stratified_split": run_stratified_split,
    "iqr_filter": run_iqr_filter,
    "missforest_train_test": run_missforest_train_test,
    "smote_balance": run_smote_balance,
    "anhydrous_normalize": run_anhydrous_normalize,
    "quantile_fit": run_quantile_fit,
    "quantile_apply": run_quantile_apply,
}
