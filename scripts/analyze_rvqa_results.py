#!/usr/bin/env python3
"""Analyze RVQA independent-run reports the same way as the notebook.

This mirrors notebooks/analyzing_rvqa_results.ipynb:
- stack report_*.xlsx files
- normalize qa_pair -> qa_id and iterfix columns
- define the error set from target != pred_no_edit
- merge wrong_edit_label from rvqa_wrong_edit_df.pkl
- report acc_wrong_edit, acc_correct_edit, and weird cases
"""

from __future__ import annotations

import argparse
import glob
import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/scratch/xxxxx/instruct_vlm_edit/indep_runs/temp_reports")
DEFAULT_WRONG_EDIT_PATH = Path("/scratch/xxxxx/rvqa_wrong_edit_df.pkl")
DEFAULT_OUTPUT_PATH = Path("/scratch/xxxxx/instruct_vlm_edit/rvqa_results_summary.txt")


def load_and_stack_xlsx(reports_path: str | Path) -> pd.DataFrame:
    files = glob.glob(os.path.join(str(reports_path), "*.xlsx"))
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No xlsx files found in {reports_path}")
    return pd.concat(dfs, ignore_index=True)


def get_reports_path(run_name: str, reports_root: Path) -> Path:
    return reports_root / run_name


def normalize_reports_df(reports_df: pd.DataFrame) -> pd.DataFrame:
    reports_df = reports_df.copy()
    reports_df.rename(columns={"qa_pair": "qa_id"}, inplace=True)
    if "qa_id" not in reports_df.columns:
        raise KeyError("Expected qa_id or qa_pair in reports")
    reports_df["indep_mode"] = reports_df.get("indep_mode", "original")
    reports_df["pred_wrong_edit_effective"] = reports_df.get("pred_wrong_edit", "")
    reports_df["gen_wrong_edit_effective"] = reports_df.get("gen_wrong_edit", "")
    reports_df["pred_wrong_edit_initial"] = reports_df.get(
        "pred_wrong_edit_initial", reports_df["pred_wrong_edit_effective"]
    )
    reports_df["gen_wrong_edit_initial"] = reports_df.get(
        "gen_wrong_edit_initial", reports_df["gen_wrong_edit_effective"]
    )
    reports_df["wrong_edit_attempts"] = reports_df.get("wrong_edit_attempts", 1)
    reports_df["wrong_edit_accepted"] = reports_df.get(
        "wrong_edit_accepted", reports_df["pred_wrong_edit_effective"] == reports_df["target"]
    )
    return reports_df


def get_error_qa_ids(reports_path: str | Path) -> set:
    reports_df = normalize_reports_df(load_and_stack_xlsx(reports_path))
    return set(reports_df.loc[reports_df["target"] != reports_df["pred_no_edit"], "qa_id"].tolist())


def bool_series(series: pd.Series) -> pd.Series:
    """Normalize booleans that may round-trip through Excel as strings."""
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    return series.map(lambda x: str(x).strip().lower() == "true").fillna(False)


def metric_line(name: str, numerator: int | float, denominator: int | float) -> str:
    pct = (float(numerator) / float(denominator) * 100.0) if denominator else 0.0
    return f"{name}: {numerator}/{denominator} = {pct:.2f}%"


def print_quality_metrics(reports_df_incorrect_pred: pd.DataFrame) -> None:
    n_error = len(reports_df_incorrect_pred)
    wrong_correct = int(bool_series(reports_df_incorrect_pred["acc_wrong_edit"]).sum())
    correct_correct = int(bool_series(reports_df_incorrect_pred["acc_correct_edit"]).sum())
    print("quality metrics on original-model error set")
    print(metric_line("acc_wrong_edit", wrong_correct, n_error))
    print(metric_line("acc_correct_edit", correct_correct, n_error))


def print_constraint_metrics(reports_df_eval: pd.DataFrame, reports_df_incorrect_pred: pd.DataFrame) -> None:
    attempts_all = pd.to_numeric(reports_df_eval["wrong_edit_attempts"], errors="coerce").fillna(1)
    attempts_error = pd.to_numeric(
        reports_df_incorrect_pred["wrong_edit_attempts"], errors="coerce"
    ).fillna(1)
    accepted_all = bool_series(reports_df_eval["wrong_edit_accepted"])
    accepted_error = bool_series(reports_df_incorrect_pred["wrong_edit_accepted"])

    print("constraint satisfaction metrics")
    print(metric_line("wrong_edit_accepted_rate_all", int(accepted_all.sum()), len(accepted_all)))
    print(
        metric_line(
            "wrong_edit_accepted_rate_error_set",
            int(accepted_error.sum()),
            len(accepted_error),
        )
    )
    print(f"avg_wrong_edit_attempts_all: {attempts_all.mean():.2f}")
    print(f"avg_wrong_edit_attempts_error_set: {attempts_error.mean():.2f}")
    print(metric_line("repair_needed_rate_all", int((attempts_all > 1).sum()), len(attempts_all)))
    print(
        metric_line(
            "repair_needed_rate_error_set",
            int((attempts_error > 1).sum()),
            len(attempts_error),
        )
    )


def prediction_reports(
    input_path: str | Path,
    wrong_edit: pd.DataFrame,
    reference_input_path: str | Path | None = None,
):
    reports_df = normalize_reports_df(load_and_stack_xlsx(input_path))
    print(f"length df: {len(reports_df)}")

    if reference_input_path is None:
        error_qa_ids = set(
            reports_df.loc[reports_df["target"] != reports_df["pred_no_edit"], "qa_id"].tolist()
        )
    else:
        error_qa_ids = get_error_qa_ids(reference_input_path)
        reports_df = reports_df[
            reports_df["qa_id"].isin(error_qa_ids) | ~reports_df["qa_id"].isin(error_qa_ids)
        ].copy()
    print(f"length error set: {len(error_qa_ids)}")

    reports_df_incorrect_pred = reports_df[reports_df["qa_id"].isin(error_qa_ids)].copy()
    reports_df_correct_pred = reports_df[~reports_df["qa_id"].isin(error_qa_ids)].copy()

    reports_df_incorrect_pred = reports_df_incorrect_pred.merge(
        wrong_edit[["qa_id", "wrong_edit_label"]], on="qa_id"
    )
    reports_df_incorrect_pred["acc_wrong_edit"] = (
        reports_df_incorrect_pred["pred_wrong_edit_effective"] == reports_df_incorrect_pred["target"]
    )
    reports_df_incorrect_pred["acc_correct_edit"] = (
        reports_df_incorrect_pred["pred_correct_edit"] == reports_df_incorrect_pred["target"]
    )
    reports_df_incorrect_pred["wrong_edit_same_no_edit_pred"] = (
        reports_df_incorrect_pred["wrong_edit_label"] == reports_df_incorrect_pred["pred_no_edit"]
    )

    acc_wrong_edit_incorrect_pred = reports_df_incorrect_pred[
        reports_df_incorrect_pred["acc_wrong_edit"] == True
    ].copy()
    acc_wrong_edit_incorrect_pred["gen_wrong_edit_parsed"] = acc_wrong_edit_incorrect_pred[
        "gen_wrong_edit_effective"
    ].apply(lambda x: str(x).strip())
    acc_wrong_edit_incorrect_pred["wrong_edit_label_parsed"] = acc_wrong_edit_incorrect_pred[
        "wrong_edit_label"
    ].apply(lambda x: str(x).strip())
    weird_cases = acc_wrong_edit_incorrect_pred[
        acc_wrong_edit_incorrect_pred["gen_wrong_edit_parsed"]
        == acc_wrong_edit_incorrect_pred["wrong_edit_label_parsed"]
    ]

    reports_df_correct_pred = reports_df_correct_pred.merge(
        wrong_edit[["qa_id", "wrong_edit_label"]], on="qa_id"
    )
    reports_df_correct_pred["acc_wrong_edit"] = (
        reports_df_correct_pred["pred_wrong_edit_effective"] == reports_df_correct_pred["target"]
    )

    return reports_df, reports_df_incorrect_pred, reports_df_correct_pred, weird_cases


def standard_run_names(reports_root: Path) -> list[str]:
    models = ["qwen3_4b", "qwen3_8b", "llava", "blip"]
    suffixes = ["original", "iterfix_1", "iterfix_2", "iterfix_3"]
    names = []
    for model in models:
        for suffix in suffixes:
            name = f"{model}_{suffix}"
            if (reports_root / name).is_dir():
                names.append(name)
    return names


def reference_for_run(run_name: str, reports_root: Path) -> Path | None:
    if "_iterfix_" not in run_name:
        return None
    model = run_name.split("_iterfix_", 1)[0]
    reference = reports_root / f"{model}_original"
    return reference if reference.is_dir() else None


def analyze_run(run_name: str, reports_root: Path, wrong_edit: pd.DataFrame) -> str:
    reports_path = get_reports_path(run_name, reports_root)
    reference_path = reference_for_run(run_name, reports_root)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=" * 80)
        print(run_name)
        print(f"reports_path: {reports_path}")
        print(f"reference_reports_path: {reference_path if reference_path else 'None'}")
        reports_df, reports_df_incorrect_pred, reports_df_correct_pred, weird_cases = prediction_reports(
            reports_path, wrong_edit, reference_input_path=reference_path
        )
        print()
        print_quality_metrics(reports_df_incorrect_pred)
        print()
        print_constraint_metrics(reports_df, reports_df_incorrect_pred)
        print()
        print('reports_df_incorrect_pred["acc_wrong_edit"].value_counts()')
        print(reports_df_incorrect_pred["acc_wrong_edit"].value_counts())
        print()
        print('reports_df_incorrect_pred["acc_correct_edit"].value_counts()')
        print(reports_df_incorrect_pred["acc_correct_edit"].value_counts())
        print()
        print("len(weird_cases)")
        print(len(weird_cases))
        print()
        print(
            'weird_cases[["pred_wrong_edit_effective", "gen_wrong_edit_effective", '
            '"wrong_edit_label", "indep_mode", "wrong_edit_attempts"]]'
        )
        cols = [
            "pred_wrong_edit_effective",
            "gen_wrong_edit_effective",
            "wrong_edit_label",
            "indep_mode",
            "wrong_edit_attempts",
        ]
        print(weird_cases[cols].to_string(index=False))
        print()
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RVQA temp report xlsx files.")
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--wrong-edit-path", type=Path, default=DEFAULT_WRONG_EDIT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--run-name", action="append", help="Specific run folder to analyze. Can repeat.")
    args = parser.parse_args()

    wrong_edit = pd.read_pickle(args.wrong_edit_path)
    wrong_edit = wrong_edit.copy()
    wrong_edit.rename(columns={"answer": "wrong_edit_label"}, inplace=True)

    run_names = args.run_name if args.run_name else standard_run_names(args.reports_root)
    if not run_names:
        raise FileNotFoundError(f"No standard run folders found in {args.reports_root}")

    parts = []
    for run_name in run_names:
        try:
            parts.append(analyze_run(run_name, args.reports_root, wrong_edit))
        except Exception as exc:
            parts.append("=" * 80 + f"\n{run_name}\nERROR: {type(exc).__name__}: {exc}\n\n")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("".join(parts))
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
