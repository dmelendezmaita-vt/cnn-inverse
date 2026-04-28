#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / "optimization_track_20260423_hh_track4_checkpoint6_shift_stability"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
RANKING_CSV = TABLES / "ranking_stability.csv"
FAMILY_METRICS_CSV = TABLES / "family_shift_metrics.csv"
SUMMARY_JSON = NOTES / "summary.json"
NOTES_MD = NOTES / "notes.md"

FAMILY_ROLE_ORDER = [
    "random_forest_clean_winner",
    "knn_robustness_winner",
    "efficientnet_neural_baseline",
]
ROLE_LABELS = {
    "random_forest_clean_winner": "random forest clean winner",
    "knn_robustness_winner": "kNN robustness winner",
    "efficientnet_neural_baseline": "EfficientNet neural baseline",
}
FAMILY_FOR_ROLE = {
    "random_forest_clean_winner": "random_forest",
    "knn_robustness_winner": "knn",
    "efficientnet_neural_baseline": "effnet",
}
ROLE_FOR_FAMILY = {value: key for key, value in FAMILY_FOR_ROLE.items()}

SHIFT_SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)(?P<sep>__|_)(?P<shift>clean|noise\d+|mult\d+|drift\d+|trunc\d+|downstep\d+|mask\d+)$"
)

# Prefer row/metrics metadata for the numeric shift value because naming is not
# perfectly consistent across the existing checkpoint4/CP4-style packages.
TOKEN_SHIFT_FALLBACKS = {
    "clean": 0.0,
    "noise001": 0.01,
    "noise002": 0.02,
    "noise005": 0.05,
    "noise010": 0.10,
    "noise020": 0.20,
}

MULT_SHIFT_BASE = 100.0
DRIFT_SHIFT_BASE = 200.0
TRUNC_SHIFT_BASE = 10000.0
DOWNSTEP_SHIFT_BASE = 20000.0
MASK_SHIFT_BASE = 30000.0


@dataclass(frozen=True)
class RunRecord:
    source_registry: str
    strategy_id: str
    identity_strategy: str
    candidate_base: str
    shift_token: str | None
    shift_value: float
    shift_label: str
    family: str
    run_dir: Path
    mae: float
    mse: float
    r2: float


@dataclass(frozen=True)
class AggregatedMetric:
    family: str
    base_strategy: str
    shift_value: float
    shift_label: str
    n_runs: int
    mae_mean: float
    mae_std: float
    mse_mean: float
    mse_std: float
    r2_mean: float
    r2_std: float
    source_registries: tuple[str, ...]
    strategy_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize classical and neural shift registries into a checkpoint6 "
            "ranking-stability package."
        )
    )
    ap.add_argument(
        "--classical-registry",
        action="append",
        required=True,
        help="Path to a classical registry CSV. May be provided multiple times.",
    )
    ap.add_argument(
        "--neural-registry",
        action="append",
        required=True,
        help="Path to a neural registry CSV. May be provided multiple times.",
    )
    ap.add_argument(
        "--out-root",
        default=str(OUT_ROOT),
        help=f"Output package root. Default: {OUT_ROOT}",
    )
    ap.add_argument(
        "--random-forest-base",
        help="Optional override for the selected random-forest base strategy.",
    )
    ap.add_argument(
        "--knn-base",
        help="Optional override for the selected kNN base strategy.",
    )
    ap.add_argument(
        "--effnet-base",
        help="Optional override for the selected EfficientNet base strategy.",
    )
    ap.add_argument(
        "--primary-metric",
        choices=["mae", "mse"],
        default="mae",
        help="Metric used for ranking stability. Default: mae.",
    )
    return ap.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO / path
    return path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def unique_fieldnames(fieldnames: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in fieldnames:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def normalize_identity_strategy(strategy_id: str) -> str:
    if strategy_id.endswith("_clean_eval"):
        return strategy_id[: -len("_clean_eval")]
    return strategy_id


def parse_shift_token(strategy_id: str) -> tuple[str, str | None]:
    if strategy_id.endswith("_clean_eval"):
        base = strategy_id[: -len("_clean_eval")]
        return base, "clean"
    match = SHIFT_SUFFIX_RE.match(strategy_id)
    if match:
        return match.group("base"), match.group("shift")
    return strategy_id, None


def infer_family(base_strategy: str) -> str | None:
    if base_strategy.startswith("random_forest"):
        return "random_forest"
    if base_strategy.startswith("knn"):
        return "knn"
    if base_strategy.startswith("effnet") or "efficientnet" in base_strategy:
        return "effnet"
    return None


def metric_value(payload: dict[str, object], key: str, split: str = "test") -> float:
    raw = payload.get(key)
    if isinstance(raw, dict):
        for candidate in [split, "test", "validate", "train"]:
            value = raw.get(candidate)
            if isinstance(value, (int, float)):
                return float(value)
        numeric_values = [float(v) for v in raw.values() if isinstance(v, (int, float))]
        if numeric_values:
            return numeric_values[0]
    elif isinstance(raw, (int, float)):
        return float(raw)
    return float("nan")


def metric_metadata_shift_value(row: dict[str, str], metrics: dict[str, object], shift_token: str | None) -> float:
    metadata = metrics.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    def _first_float(*values: object) -> float | None:
        for value in values:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str) and value:
                return float(value)
        return None

    additive = _first_float(
        row.get("features_additive_noise_std"),
        row.get("noise_std"),
        row.get("shift_std"),
        metadata.get("features_additive_noise_std"),
        metadata.get("noise_std"),
        metadata.get("shift_std"),
    )
    multiplicative = _first_float(
        row.get("features_multiplicative_noise_std"),
        metadata.get("features_multiplicative_noise_std"),
    )
    drift = _first_float(
        row.get("features_baseline_drift_std"),
        metadata.get("features_baseline_drift_std"),
    )
    mask = _first_float(
        row.get("features_mask_fraction"),
        metadata.get("features_mask_fraction"),
    )
    sub_step = _first_float(
        row.get("features_sub_step"),
        metadata.get("features_sub_step"),
    )
    sub_length = _first_float(
        row.get("features_sub_length"),
        metadata.get("features_sub_length"),
    )

    if mask is not None and mask > 0.0:
        return MASK_SHIFT_BASE + mask
    if sub_step is not None and sub_step > 1.0:
        return DOWNSTEP_SHIFT_BASE + sub_step
    if sub_length is not None and sub_length > 0.0 and int(round(sub_length)) != 2000:
        return TRUNC_SHIFT_BASE + float(int(round(sub_length)))
    if drift is not None and drift > 0.0:
        return DRIFT_SHIFT_BASE + drift
    if multiplicative is not None and multiplicative > 0.0:
        return MULT_SHIFT_BASE + multiplicative
    if additive is not None:
        return additive

    if shift_token is None:
        return float("nan")
    if shift_token in TOKEN_SHIFT_FALLBACKS:
        return TOKEN_SHIFT_FALLBACKS[shift_token]
    if shift_token.startswith("mult"):
        return MULT_SHIFT_BASE + (float(shift_token[4:]) / 100.0)
    if shift_token.startswith("drift"):
        return DRIFT_SHIFT_BASE + (float(shift_token[5:]) / 100.0)
    if shift_token.startswith("trunc"):
        return TRUNC_SHIFT_BASE + float(int(shift_token[5:]))
    if shift_token.startswith("downstep"):
        return DOWNSTEP_SHIFT_BASE + float(int(shift_token[8:]))
    if shift_token.startswith("mask"):
        return MASK_SHIFT_BASE + (float(shift_token[4:]) / 100.0)
    return float("nan")


def canonical_shift_label(shift_value: float) -> str:
    if math.isnan(shift_value) or abs(shift_value) < 1e-12:
        return "clean"
    if shift_value >= MASK_SHIFT_BASE:
        frac = shift_value - MASK_SHIFT_BASE
        return f"mask{int(round(frac * 100)):02d}"
    if shift_value >= DOWNSTEP_SHIFT_BASE:
        step = int(round(shift_value - DOWNSTEP_SHIFT_BASE))
        return f"downstep{step}"
    if shift_value >= TRUNC_SHIFT_BASE:
        length = int(round(shift_value - TRUNC_SHIFT_BASE))
        return f"trunc{length}"
    if shift_value >= DRIFT_SHIFT_BASE:
        std = shift_value - DRIFT_SHIFT_BASE
        return f"drift{int(round(std * 100)):03d}"
    if shift_value >= MULT_SHIFT_BASE:
        std = shift_value - MULT_SHIFT_BASE
        return f"mult{int(round(std * 100)):03d}"
    return f"noise_std_{shift_value:.3f}"


def shift_key(shift_value: float) -> float:
    if math.isnan(shift_value):
        return float("nan")
    return round(float(shift_value), 6)


def resolve_metrics_dir(run_output_root: str) -> Path:
    root = resolve_path(run_output_root)
    candidates: list[Path] = []
    for probe in [root, root / "run_dnn"]:
        if (probe / "metrics_summary.json").exists():
            return probe
    if not root.exists():
        raise FileNotFoundError(f"Missing run_output_root: {root}")
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        for probe in [child, child / "run_dnn"]:
            if (probe / "metrics_summary.json").exists():
                candidates.append(probe)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No metrics_summary.json found under {root}")


def candidate_base_multi_shift_map(records: list[RunRecord]) -> set[str]:
    shifts_by_candidate: dict[str, set[float]] = defaultdict(set)
    for record in records:
        shifts_by_candidate[record.candidate_base].add(record.shift_value)
    return {base for base, shifts in shifts_by_candidate.items() if len(shifts) >= 2}


def finalize_base_strategy(records: list[RunRecord]) -> list[RunRecord]:
    multi_shift_bases = candidate_base_multi_shift_map(records)
    finalized: list[RunRecord] = []
    for record in records:
        final_base = record.candidate_base if record.candidate_base in multi_shift_bases else record.identity_strategy
        finalized.append(
            RunRecord(
                source_registry=record.source_registry,
                strategy_id=record.strategy_id,
                identity_strategy=record.identity_strategy,
                candidate_base=final_base,
                shift_token=record.shift_token,
                shift_value=record.shift_value,
                shift_label=record.shift_label,
                family=record.family,
                run_dir=record.run_dir,
                mae=record.mae,
                mse=record.mse,
                r2=record.r2,
            )
        )
    return finalized


def load_registry_records(registry_paths: list[Path]) -> tuple[list[RunRecord], list[str]]:
    records: list[RunRecord] = []
    warnings: list[str] = []
    seen_runs: set[tuple[str, str]] = set()

    for registry_path in registry_paths:
        rows = load_rows(registry_path)
        for row in rows:
            status = (row.get("status") or "COMPLETED").strip().upper()
            if status != "COMPLETED":
                continue
            strategy_id = row.get("strategy_id", "").strip()
            run_output_root = row.get("run_output_root", "").strip()
            if not strategy_id or not run_output_root:
                warnings.append(
                    f"Skipped row from {registry_path.name}: missing strategy_id or run_output_root."
                )
                continue
            try:
                metrics_dir = resolve_metrics_dir(run_output_root)
                metrics = json.loads((metrics_dir / "metrics_summary.json").read_text())
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"Skipped {strategy_id} from {registry_path.name}: {exc}"
                )
                continue

            identity_strategy = normalize_identity_strategy(strategy_id)
            candidate_base, shift_token = parse_shift_token(strategy_id)
            family = infer_family(candidate_base)
            if family is None:
                family = infer_family(identity_strategy)
            if family is None:
                continue

            shift_value = metric_metadata_shift_value(row, metrics, shift_token)
            if math.isnan(shift_value):
                warnings.append(
                    f"Skipped {strategy_id} from {registry_path.name}: could not infer shift value."
                )
                continue
            label = canonical_shift_label(shift_value)

            mae = metric_value(metrics, "mae")
            mse = metric_value(metrics, "mse")
            r2 = metric_value(metrics, "r2")
            if math.isnan(mae):
                warnings.append(
                    f"Skipped {strategy_id} from {registry_path.name}: missing test MAE."
                )
                continue

            dedupe_key = (strategy_id, str(metrics_dir))
            if dedupe_key in seen_runs:
                continue
            seen_runs.add(dedupe_key)
            records.append(
                RunRecord(
                    source_registry=str(registry_path),
                    strategy_id=strategy_id,
                    identity_strategy=identity_strategy,
                    candidate_base=candidate_base,
                    shift_token=shift_token,
                    shift_value=shift_key(shift_value),
                    shift_label=label,
                    family=family,
                    run_dir=metrics_dir,
                    mae=mae,
                    mse=mse,
                    r2=r2,
                )
            )

    return finalize_base_strategy(records), warnings


def aggregate_records(records: list[RunRecord]) -> list[AggregatedMetric]:
    grouped: dict[tuple[str, str, float], list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.family, record.candidate_base, record.shift_value)].append(record)

    aggregated: list[AggregatedMetric] = []
    for (family, base_strategy, shift_value), items in sorted(grouped.items()):
        mae_values = [item.mae for item in items]
        mse_values = [item.mse for item in items if not math.isnan(item.mse)]
        r2_values = [item.r2 for item in items if not math.isnan(item.r2)]
        aggregated.append(
            AggregatedMetric(
                family=family,
                base_strategy=base_strategy,
                shift_value=shift_value,
                shift_label=items[0].shift_label,
                n_runs=len(items),
                mae_mean=mean(mae_values),
                mae_std=pstdev(mae_values) if len(mae_values) > 1 else 0.0,
                mse_mean=mean(mse_values) if mse_values else float("nan"),
                mse_std=pstdev(mse_values) if len(mse_values) > 1 else 0.0,
                r2_mean=mean(r2_values) if r2_values else float("nan"),
                r2_std=pstdev(r2_values) if len(r2_values) > 1 else 0.0,
                source_registries=tuple(sorted({item.source_registry for item in items})),
                strategy_ids=tuple(sorted({item.strategy_id for item in items})),
            )
        )
    return aggregated


def grouped_family_metrics(aggregated: list[AggregatedMetric], family: str) -> dict[str, dict[float, AggregatedMetric]]:
    out: dict[str, dict[float, AggregatedMetric]] = defaultdict(dict)
    for item in aggregated:
        if item.family == family:
            out[item.base_strategy][item.shift_value] = item
    return out


def rank_candidates_by_shift(
    candidate_metrics: dict[str, dict[float, AggregatedMetric]],
    shift_values: list[float],
    metric_name: str,
) -> dict[str, tuple[float, float, int]]:
    rank_summaries: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for shift_value in shift_values:
        available = [
            (base_strategy, metrics[shift_value])
            for base_strategy, metrics in candidate_metrics.items()
            if shift_value in metrics
        ]
        available.sort(key=lambda item: (getattr(item[1], metric_name), item[0]))
        for rank, (base_strategy, item) in enumerate(available, start=1):
            rank_summaries[base_strategy].append((rank, getattr(item, metric_name)))
    scored: dict[str, tuple[float, float, int]] = {}
    for base_strategy, values in rank_summaries.items():
        mean_rank = mean([rank for rank, _ in values])
        mean_metric = mean([metric for _, metric in values])
        scored[base_strategy] = (mean_rank, mean_metric, len(values))
    return scored


def select_base_strategy(
    family: str,
    aggregated: list[AggregatedMetric],
    primary_metric: str,
    override: str | None = None,
) -> tuple[str, str]:
    candidates = grouped_family_metrics(aggregated, family)
    if not candidates:
        raise RuntimeError(f"No aggregated rows found for family={family}")

    if override:
        if override not in candidates:
            available = ", ".join(sorted(candidates))
            raise RuntimeError(
                f"Override {override!r} not found for family={family}. Available: {available}"
            )
        return override, "user override"

    if family == "random_forest":
        clean_candidates = [
            (base_strategy, metrics[0.0])
            for base_strategy, metrics in candidates.items()
            if 0.0 in metrics
        ]
        if clean_candidates:
            clean_candidates.sort(key=lambda item: (getattr(item[1], primary_metric + "_mean"), item[0]))
            best = clean_candidates[0][0]
            return best, f"best clean {primary_metric.upper()} among {len(clean_candidates)} random-forest candidates"

    non_clean_shifts = sorted(
        {
            shift_value
            for metrics in candidates.values()
            for shift_value in metrics
            if abs(shift_value) > 1e-12
        }
    )
    if family == "knn" and non_clean_shifts:
        scored = rank_candidates_by_shift(candidates, non_clean_shifts, primary_metric + "_mean")
        best = sorted(
            scored.items(),
            key=lambda item: (-item[1][2], item[1][0], item[1][1], item[0]),
        )[0][0]
        return best, f"best mean noisy-shift rank across {len(non_clean_shifts)} shifts"

    clean_candidates = [
        (base_strategy, metrics[0.0])
        for base_strategy, metrics in candidates.items()
        if 0.0 in metrics
    ]
    if clean_candidates:
        clean_candidates.sort(key=lambda item: (getattr(item[1], primary_metric + "_mean"), item[0]))
        best = clean_candidates[0][0]
        return best, f"best clean {primary_metric.upper()} among {len(clean_candidates)} {family} candidates"

    all_shifts = sorted({shift_value for metrics in candidates.values() for shift_value in metrics})
    scored = rank_candidates_by_shift(candidates, all_shifts, primary_metric + "_mean")
    best = sorted(
        scored.items(),
        key=lambda item: (-item[1][2], item[1][0], item[1][1], item[0]),
    )[0][0]
    return best, f"best mean rank across {len(all_shifts)} available shifts"


def family_metric_rows(
    selected: dict[str, tuple[str, str]],
    aggregated: list[AggregatedMetric],
) -> tuple[list[dict[str, object]], dict[str, dict[float, AggregatedMetric]]]:
    lookup: dict[str, dict[float, AggregatedMetric]] = {}
    rows: list[dict[str, object]] = []
    by_key = {
        (item.family, item.base_strategy, item.shift_value): item
        for item in aggregated
    }
    for role in FAMILY_ROLE_ORDER:
        family = FAMILY_FOR_ROLE[role]
        base_strategy, selection_reason = selected[role]
        family_lookup: dict[float, AggregatedMetric] = {}
        for item in aggregated:
            if item.family == family and item.base_strategy == base_strategy:
                family_lookup[item.shift_value] = item
        lookup[role] = family_lookup
        for shift_value, item in sorted(family_lookup.items()):
            rows.append(
                {
                    "family_role": role,
                    "family_label": ROLE_LABELS[role],
                    "family": family,
                    "base_strategy": base_strategy,
                    "selection_reason": selection_reason,
                    "shift_label": item.shift_label,
                    "shift_value": item.shift_value,
                    "n_runs": item.n_runs,
                    "mae_mean": item.mae_mean,
                    "mae_std": item.mae_std,
                    "mse_mean": item.mse_mean,
                    "mse_std": item.mse_std,
                    "r2_mean": item.r2_mean,
                    "r2_std": item.r2_std,
                    "strategy_ids": ";".join(item.strategy_ids),
                    "source_registries": ";".join(item.source_registries),
                }
            )
    return rows, lookup


def pairwise_flip_count(reference_order: list[str], current_order: list[str]) -> int:
    ref_pos = {role: idx for idx, role in enumerate(reference_order)}
    current_pos = {role: idx for idx, role in enumerate(current_order)}
    flips = 0
    for idx, left in enumerate(reference_order):
        for right in reference_order[idx + 1 :]:
            if (ref_pos[left] - ref_pos[right]) * (current_pos[left] - current_pos[right]) < 0:
                flips += 1
    return flips


def build_ranking_rows(
    family_lookup: dict[str, dict[float, AggregatedMetric]],
    selected: dict[str, tuple[str, str]],
    primary_metric: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    common_shifts = sorted(
        set.intersection(*(set(items.keys()) for items in family_lookup.values()))
    )
    if not common_shifts:
        raise RuntimeError("No common shifts exist across the selected random-forest, kNN, and EfficientNet representatives")

    reference_shift = 0.0 if 0.0 in common_shifts else common_shifts[0]
    reference_ranked = sorted(
        FAMILY_ROLE_ORDER,
        key=lambda role: (getattr(family_lookup[role][reference_shift], primary_metric + "_mean"), role),
    )

    rows: list[dict[str, object]] = []
    winner_counts: dict[str, int] = defaultdict(int)
    stable_count = 0
    unstable_shifts: list[str] = []

    for shift_value in common_shifts:
        ranked_roles = sorted(
            FAMILY_ROLE_ORDER,
            key=lambda role: (getattr(family_lookup[role][shift_value], primary_metric + "_mean"), role),
        )
        flips = pairwise_flip_count(reference_ranked, ranked_roles)
        stable = flips == 0
        if stable:
            stable_count += 1
        else:
            unstable_shifts.append(canonical_shift_label(shift_value))
        winner_counts[ranked_roles[0]] += 1

        winner_item = family_lookup[ranked_roles[0]][shift_value]
        runner_up_item = family_lookup[ranked_roles[1]][shift_value]
        winner_metric = getattr(winner_item, primary_metric + "_mean")
        runner_up_metric = getattr(runner_up_item, primary_metric + "_mean")
        margin = runner_up_metric - winner_metric
        margin_pct = (margin / runner_up_metric) if runner_up_metric else float("nan")

        row: dict[str, object] = {
            "shift_label": canonical_shift_label(shift_value),
            "shift_value": shift_value,
            "reference_shift_label": canonical_shift_label(reference_shift),
            "rank_signature": " > ".join(ranked_roles),
            "reference_rank_signature": " > ".join(reference_ranked),
            "winner_family_role": ranked_roles[0],
            "winner_strategy_base": selected[ranked_roles[0]][0],
            "winner_margin_vs_runner_up": margin,
            "winner_margin_pct_vs_runner_up": margin_pct,
            "pairwise_flips_vs_reference": flips,
            "stable_vs_reference": stable,
            "winner_changed_vs_reference": ranked_roles[0] != reference_ranked[0],
        }
        for rank, role in enumerate(ranked_roles, start=1):
            item = family_lookup[role][shift_value]
            row[f"rank_{rank}_family_role"] = role
            row[f"rank_{rank}_strategy_base"] = selected[role][0]
            row[f"rank_{rank}_{primary_metric}_mean"] = getattr(item, primary_metric + "_mean")
        for role in FAMILY_ROLE_ORDER:
            item = family_lookup[role][shift_value]
            row[f"{role}_strategy_base"] = selected[role][0]
            row[f"{role}_{primary_metric}_mean"] = getattr(item, primary_metric + "_mean")
            row[f"{role}_mae_mean"] = item.mae_mean
            row[f"{role}_mse_mean"] = item.mse_mean
            row[f"{role}_r2_mean"] = item.r2_mean
            row[f"{role}_rank"] = ranked_roles.index(role) + 1
            row[f"{role}_n_runs"] = item.n_runs
        rows.append(row)

    summary = {
        "reference_shift_label": canonical_shift_label(reference_shift),
        "reference_rank_signature": " > ".join(reference_ranked),
        "common_shift_labels": [canonical_shift_label(value) for value in common_shifts],
        "stable_shift_count": stable_count,
        "unstable_shift_count": len(common_shifts) - stable_count,
        "unstable_shift_labels": unstable_shifts,
        "winner_counts": {role: winner_counts.get(role, 0) for role in FAMILY_ROLE_ORDER},
    }
    return rows, summary


def build_notes(
    classical_registries: list[Path],
    neural_registries: list[Path],
    selected: dict[str, tuple[str, str]],
    ranking_summary: dict[str, object],
    ranking_rows: list[dict[str, object]],
    warnings: list[str],
    out_root: Path,
) -> str:
    note_lines = [
        "# HH Track4 Checkpoint6 Shift Stability",
        "",
        f"- package root: `{out_root}`",
        f"- classical registries: {', '.join(str(path) for path in classical_registries)}",
        f"- neural registries: {', '.join(str(path) for path in neural_registries)}",
        f"- ranking table: `{RANKING_CSV.name}`",
        f"- family metrics: `{FAMILY_METRICS_CSV.name}`",
        f"- summary json: `{SUMMARY_JSON.name}`",
        "",
        "## Selected Representatives",
    ]
    for role in FAMILY_ROLE_ORDER:
        base_strategy, selection_reason = selected[role]
        note_lines.append(
            f"- {ROLE_LABELS[role]}: `{base_strategy}` ({selection_reason})"
        )

    note_lines.extend(
        [
            "",
            "## Short Readout",
            f"- reference shift: `{ranking_summary['reference_shift_label']}`",
            f"- reference ranking: `{ranking_summary['reference_rank_signature']}`",
            f"- compared common shifts: {', '.join(f'`{label}`' for label in ranking_summary['common_shift_labels'])}",
            f"- stable shifts: `{ranking_summary['stable_shift_count']}` / `{len(ranking_rows)}`",
            f"- unstable shifts: {', '.join(f'`{label}`' for label in ranking_summary['unstable_shift_labels']) if ranking_summary['unstable_shift_labels'] else '`none`'}",
            "- winner counts: "
            + ", ".join(
                f"`{role}`={ranking_summary['winner_counts'][role]}"
                for role in FAMILY_ROLE_ORDER
            ),
        ]
    )

    if warnings:
        note_lines.extend(["", "## Warnings"])
        for warning in warnings[:10]:
            note_lines.append(f"- {warning}")
        if len(warnings) > 10:
            note_lines.append(f"- plus `{len(warnings) - 10}` additional skipped-row warnings")

    note_lines.append("")
    return "\n".join(note_lines) + "\n"


def main() -> None:
    args = parse_args()

    classical_registries = [resolve_path(path) for path in args.classical_registry]
    neural_registries = [resolve_path(path) for path in args.neural_registry]
    out_root = resolve_path(args.out_root)
    tables = out_root / "tables"
    notes = out_root / "notes"
    ranking_csv = tables / RANKING_CSV.name
    family_metrics_csv = tables / FAMILY_METRICS_CSV.name
    summary_json = notes / SUMMARY_JSON.name
    notes_md = notes / NOTES_MD.name

    tables.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)

    classical_records, classical_warnings = load_registry_records(classical_registries)
    neural_records, neural_warnings = load_registry_records(neural_registries)
    all_records = classical_records + neural_records
    warnings = classical_warnings + neural_warnings
    if not all_records:
        raise RuntimeError("No completed shift records were loaded from the provided registries")

    aggregated = aggregate_records(all_records)

    selected = {
        "random_forest_clean_winner": select_base_strategy(
            "random_forest", aggregated, args.primary_metric, args.random_forest_base
        ),
        "knn_robustness_winner": select_base_strategy(
            "knn", aggregated, args.primary_metric, args.knn_base
        ),
        "efficientnet_neural_baseline": select_base_strategy(
            "effnet", aggregated, args.primary_metric, args.effnet_base
        ),
    }

    family_rows, family_lookup = family_metric_rows(selected, aggregated)
    ranking_rows, ranking_summary = build_ranking_rows(family_lookup, selected, args.primary_metric)

    write_csv(
        family_metrics_csv,
        family_rows,
        [
            "family_role",
            "family_label",
            "family",
            "base_strategy",
            "selection_reason",
            "shift_label",
            "shift_value",
            "n_runs",
            "mae_mean",
            "mae_std",
            "mse_mean",
            "mse_std",
            "r2_mean",
            "r2_std",
            "strategy_ids",
            "source_registries",
        ],
    )

    ranking_fieldnames = [
        "shift_label",
        "shift_value",
        "reference_shift_label",
        "rank_signature",
        "reference_rank_signature",
        "winner_family_role",
        "winner_strategy_base",
        "winner_margin_vs_runner_up",
        "winner_margin_pct_vs_runner_up",
        "pairwise_flips_vs_reference",
        "stable_vs_reference",
        "winner_changed_vs_reference",
    ]
    for rank in [1, 2, 3]:
        ranking_fieldnames.extend(
            [
                f"rank_{rank}_family_role",
                f"rank_{rank}_strategy_base",
                f"rank_{rank}_{args.primary_metric}_mean",
            ]
        )
    for role in FAMILY_ROLE_ORDER:
        ranking_fieldnames.extend(
            [
                f"{role}_strategy_base",
                f"{role}_{args.primary_metric}_mean",
                f"{role}_mae_mean",
                f"{role}_mse_mean",
                f"{role}_r2_mean",
                f"{role}_rank",
                f"{role}_n_runs",
            ]
        )
    write_csv(ranking_csv, ranking_rows, unique_fieldnames(ranking_fieldnames))

    summary_payload = {
        "classical_registries": [str(path) for path in classical_registries],
        "neural_registries": [str(path) for path in neural_registries],
        "selected_representatives": {
            role: {
                "base_strategy": selected[role][0],
                "selection_reason": selected[role][1],
            }
            for role in FAMILY_ROLE_ORDER
        },
        "ranking_summary": ranking_summary,
        "warnings": warnings,
    }
    write_json(summary_json, summary_payload)

    notes_text = build_notes(
        classical_registries=classical_registries,
        neural_registries=neural_registries,
        selected=selected,
        ranking_summary=ranking_summary,
        ranking_rows=ranking_rows,
        warnings=warnings,
        out_root=out_root,
    )
    notes_md.write_text(notes_text)

    print(f"Wrote ranking table: {ranking_csv}")
    print(f"Wrote family metrics: {family_metrics_csv}")
    print(f"Wrote summary json: {summary_json}")
    print(f"Wrote notes: {notes_md}")
    for role in FAMILY_ROLE_ORDER:
        print(f"{role}: {selected[role][0]} ({selected[role][1]})")


if __name__ == "__main__":
    main()
