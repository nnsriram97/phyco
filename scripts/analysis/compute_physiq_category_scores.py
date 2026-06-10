#!/usr/bin/env python3
"""
Compute Physics-IQ scores per category using Physics-IQ benchmark outputs.

This script mirrors the official scoring logic and aggregates by category
derived from descriptions.csv. Results are printed and written to CSV.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

VIEWS: Sequence[str] = ("perspective-left", "perspective-center", "perspective-right")


def parse_list_of_floats(value):
    """Parse stringified float lists into rounded float lists."""
    if isinstance(value, list):
        return [round(float(x), 4) for x in value]
    if isinstance(value, str):
        if not value.strip():
            return []
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", value)
        return [round(float(x), 4) for x in numbers]
    raise TypeError(f"Unsupported type for list parsing: {type(value)}")


def _mean_concatenated(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    arrays = [np.asarray(lst, dtype=float) for lst in series if isinstance(lst, list) and lst]
    if not arrays:
        return float("nan")
    concatenated = np.concatenate(arrays)
    return float(np.mean(concatenated))


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    list_cols = []
    for view in VIEWS:
        list_cols.extend(
            [
                f"v1_mse_{view}",
                f"spatiotemporal_iou_v1_{view}",
                f"variance_mse_{view}",
                f"variance_spatiotemporal_iou_{view}",
            ]
        )
    existing_cols = [c for c in list_cols if c in df.columns]
    for col in existing_cols:
        df[col] = df[col].apply(parse_list_of_floats)
    return df


def calculate_iq_from_subset(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty:
        return float("nan"), float("nan")

    total_sum_v1_mse = (
        df.apply(lambda row: np.mean(np.concatenate([row[f"v1_mse_{view}"] for view in VIEWS])), axis=1).mean()
    )
    total_sum_spatiotemporal_iou_v1 = (
        df.apply(
            lambda row: np.mean(np.concatenate([row[f"spatiotemporal_iou_v1_{view}"] for view in VIEWS])),
            axis=1,
        ).mean()
    )

    total_sum_spatial_iou = df[[f"spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()
    total_sum_weighted_spatial_iou = df[[f"weighted_spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()

    variance_mse_values: List[float] = []
    variance_spatiotemporal_values: List[float] = []

    for view in VIEWS:
        variance_mse_col = f"variance_mse_{view}"
        if variance_mse_col in df.columns:
            variance_mse_values.append(_mean_concatenated(df[variance_mse_col]))
        variance_st_col = f"variance_spatiotemporal_iou_{view}"
        if variance_st_col in df.columns:
            variance_spatiotemporal_values.append(_mean_concatenated(df[variance_st_col]))

    physical_variance_mse = float(np.nanmean(variance_mse_values)) if variance_mse_values else float("nan")
    physical_variance_spatiotemporal_iou = (
        float(np.nanmean(variance_spatiotemporal_values)) if variance_spatiotemporal_values else float("nan")
    )

    physical_variance_spatial = float(
        np.mean([df[f"variance_spatial_{view}"].mean() for view in VIEWS if f"variance_spatial_{view}" in df.columns])
    )
    physical_variance_weighted_spatial = float(
        np.mean(
            [
                df[f"variance_weighted_spatial_{view}"].mean()
                for view in VIEWS
                if f"variance_weighted_spatial_{view}" in df.columns
            ]
        )
    )

    physical_variance_all_metrics = round(
        physical_variance_spatiotemporal_iou + physical_variance_spatial + physical_variance_weighted_spatial - physical_variance_mse,
        4,
    )

    final_score = (
        (
            (total_sum_spatiotemporal_iou_v1 / physical_variance_spatiotemporal_iou)
            + (total_sum_spatial_iou / physical_variance_spatial)
            + (total_sum_weighted_spatial_iou / physical_variance_weighted_spatial)
        )
        / 3
    ) - (total_sum_v1_mse - physical_variance_mse)

    final_score *= 100
    final_score = round(max(min(final_score, 100.0), 0.0), 2)

    return final_score, physical_variance_all_metrics


def normalize_name(name: str) -> str:
    return Path(name).name


def candidate_keys(name: str) -> set[str]:
    keys: set[str] = set()
    if not isinstance(name, str) or not name:
        return keys
    keys.add(name)
    base = normalize_name(name)
    keys.add(base)
    parts = base.split("_")
    if len(parts) > 1:
        for i in range(1, len(parts)):
            suffix = "_".join(parts[i:])
            keys.add(suffix)
    return keys


def load_categories(descriptions_csv: Path) -> dict[str, str]:
    desc_df = pd.read_csv(descriptions_csv)
    if "scenario" not in desc_df.columns or "category" not in desc_df.columns:
        raise ValueError("descriptions.csv must contain 'scenario' and 'category' columns.")

    categories: dict[str, str] = {}
    for _, row in desc_df.iterrows():
        category = row["category"]
        scenario = row["scenario"]
        generated_name = row.get("generated_video_name")
        for key in candidate_keys(scenario):
            categories[key] = category
        if isinstance(generated_name, str) and generated_name:
            for key in candidate_keys(generated_name):
                categories[key] = category
    return categories


def compute_category_scores(results_csv: Path, descriptions_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df = prepare_dataframe(df)
    category_map = load_categories(descriptions_csv)
    def lookup_category(name: str) -> str | None:
        for key in candidate_keys(name):
            cat = category_map.get(key)
            if cat:
                return cat
        return None

    df["category"] = df["scenario"].apply(lookup_category)
    if df["category"].isna().any():
        missing = df[df["category"].isna()]["scenario"].tolist()
        print(f"[WARN] {len(missing)} scenarios missing category labels; they will be skipped.")
        df = df.dropna(subset=["category"])

    rows = []
    for category, group in df.groupby("category"):
        score, variance = calculate_iq_from_subset(group)
        rows.append(
            {
                "category": category,
                "num_scenarios": len(group),
                "physics_iq_score": score,
                "physical_variance": variance,
            }
        )

    if not rows:
        result_df = pd.DataFrame(columns=["category", "num_scenarios", "physics_iq_score", "physical_variance"])
    else:
        result_df = pd.DataFrame(rows).sort_values(by="physics_iq_score", ascending=False)
    result_df.to_csv(output_csv, index=False)
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Physics-IQ scores per category.")
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=Path("/net/acadia1a/data/sriram/physics-IQ-benchmark/output/physics-IQ-benchmark/results/physics-IQ-benchmark.csv"),
        help="Path to physics-IQ benchmark results CSV.",
    )
    parser.add_argument(
        "--descriptions_csv",
        type=Path,
        default=Path("/net/acadia1a/data/sriram/physics-IQ-benchmark/descriptions/descriptions.csv"),
        help="Path to descriptions.csv containing scenario categories.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <results_dir>/physics-IQ-benchmark_category_scores.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = args.results_csv.with_name(args.results_csv.stem + "_category_scores.csv")

    result_df = compute_category_scores(args.results_csv, args.descriptions_csv, output_csv)
    print("Per-category Physics-IQ scores:")
    print(result_df.to_string(index=False))
    print(f"\nSaved category scores to {output_csv}")


if __name__ == "__main__":
    main()
