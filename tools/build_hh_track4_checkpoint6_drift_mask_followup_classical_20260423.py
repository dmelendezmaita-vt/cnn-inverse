#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

DATE_TAG = "20260423_hh_track4_checkpoint6_drift_mask_followup_classical"
DATE_LABEL = "2026-04-23"

REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / f"optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
CONFIG_ROOT = REPO / f"src/pytorch/configs/reruns_{DATE_TAG}"

MATRIX_CSV = TABLES / f"hh_track4_checkpoint6_drift_mask_followup_classical_matrix_{DATE_TAG}.csv"
NOTES_MD = NOTES / f"hh_track4_checkpoint6_drift_mask_followup_classical_notes_{DATE_TAG}.md"


def read_text(path: Path) -> str:
    return path.read_text()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def create_config(base_rel: str, out_rel: str, *, seed: int, shift: dict[str, str]) -> str:
    base_path = REPO / base_rel
    out_path = REPO / out_rel
    params = yaml.safe_load(read_text(base_path))

    params.setdefault("data", {})
    params["data"]["features_normalize"] = True
    params["data"]["features_scale_cache_enabled"] = True
    params["data"]["split_array_cache_enabled"] = True
    params["data"]["dataloader_num_workers"] = 0
    params["data"]["dataloader_persistent_workers"] = False
    params["data"]["dataloader_prefetch_factor"] = None
    params["data"]["dataloader_pin_memory"] = False
    params["data"]["features_sub_begin_random"] = False
    params["data"]["features_sub_begin_random_eval"] = False
    params["data"]["features_sub_length"] = int(shift["features_sub_length"])
    params["data"]["features_sub_step"] = int(shift["features_sub_step"])
    params["data"]["features_additive_noise_std"] = float(shift["features_additive_noise_std"])
    params["data"]["features_multiplicative_noise_std"] = float(shift["features_multiplicative_noise_std"])
    params["data"]["features_baseline_drift_std"] = float(shift["features_baseline_drift_std"])
    params["data"]["features_mask_fraction"] = float(shift["features_mask_fraction"])
    params["data"]["random_seed"] = seed
    params["data"]["Ntrain"] = 4096
    params["data"]["Nvalidate"] = 1024
    params["data"]["Ntest"] = 1024
    params.setdefault("runconfig", {})
    params["runconfig"]["save_predictions"] = "test"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(params, sort_keys=False))
    return out_rel


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    track_key = "track4_hh_full"
    meta = {
        "short": "t4cp6dmc",
        "base": "src/pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml",
        "tar": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
        "prefix": "concatenated_data",
        "shared": str(
            IMPORTANT
            / "optimization_track_20260411_v100_interactive"
            / "shared_data"
            / "track4_hh_full"
            / "concatenated_data"
        ),
    }

    baselines = {
        "random_forest_500": "Checkpoint6 clean-data classical winner.",
        "knn_k11": "Checkpoint6 robustness classical winner.",
    }
    feature_mode = "raw_plus_fft256_summary12"
    shifts = [
        ("clean", 1240, {
            "features_additive_noise_std": "0.0",
            "features_multiplicative_noise_std": "0.0",
            "features_baseline_drift_std": "0.0",
            "features_mask_fraction": "0.0",
            "features_sub_length": "2000",
            "features_sub_step": "1",
        }, "Clean reference."),
        ("drift005", 1241, {
            "features_additive_noise_std": "0.0",
            "features_multiplicative_noise_std": "0.0",
            "features_baseline_drift_std": "0.05",
            "features_mask_fraction": "0.0",
            "features_sub_length": "2000",
            "features_sub_step": "1",
        }, "Linear baseline drift, std=0.05."),
        ("drift010", 1242, {
            "features_additive_noise_std": "0.0",
            "features_multiplicative_noise_std": "0.0",
            "features_baseline_drift_std": "0.10",
            "features_mask_fraction": "0.0",
            "features_sub_length": "2000",
            "features_sub_step": "1",
        }, "Linear baseline drift, std=0.10."),
        ("drift020", 1243, {
            "features_additive_noise_std": "0.0",
            "features_multiplicative_noise_std": "0.0",
            "features_baseline_drift_std": "0.20",
            "features_mask_fraction": "0.0",
            "features_sub_length": "2000",
            "features_sub_step": "1",
        }, "Linear baseline drift, std=0.20."),
        ("mask10", 1244, {
            "features_additive_noise_std": "0.0",
            "features_multiplicative_noise_std": "0.0",
            "features_baseline_drift_std": "0.0",
            "features_mask_fraction": "0.10",
            "features_sub_length": "2000",
            "features_sub_step": "1",
        }, "Contiguous missing segment covering 10% of the trace."),
        ("mask20", 1245, {
            "features_additive_noise_std": "0.0",
            "features_multiplicative_noise_std": "0.0",
            "features_baseline_drift_std": "0.0",
            "features_mask_fraction": "0.20",
            "features_sub_length": "2000",
            "features_sub_step": "1",
        }, "Contiguous missing segment covering 20% of the trace."),
        ("mask30", 1246, {
            "features_additive_noise_std": "0.0",
            "features_multiplicative_noise_std": "0.0",
            "features_baseline_drift_std": "0.0",
            "features_mask_fraction": "0.30",
            "features_sub_length": "2000",
            "features_sub_step": "1",
        }, "Contiguous missing segment covering 30% of the trace."),
    ]

    rows = []
    row_num = 0
    for baseline_name, baseline_desc in baselines.items():
        group = f"track4_checkpoint6_drift_mask_{baseline_name}"
        for shift_name, seed, shift_cfg, shift_desc in shifts:
            row_num += 1
            cfg_rel = (
                f"src/pytorch/configs/reruns_{DATE_TAG}/{track_key}/"
                f"params_{meta['short']}_{baseline_name}_{shift_name}_n1_s{seed}.yaml"
            )
            create_config(meta["base"], cfg_rel, seed=seed, shift=shift_cfg)
            strategy_id = f"{baseline_name}__{shift_name}"
            rows.append(
                {
                    "row_id": f"cp6dmc_{row_num:04d}",
                    "phase": "phaseCP6_track4_drift_mask_followup_classical",
                    "phase_label": "Checkpoint6 drift/mask follow-up classical",
                    "launch_mode": "serial_1x1n",
                    "launch_group": group,
                    "family": "track4_checkpoint6_drift_mask_followup_classical",
                    "track_key": track_key,
                    "policy": "strong",
                    "nodes": "1",
                    "cpus_per_node": "18",
                    "seed": str(seed),
                    "strategy_id": strategy_id,
                    "strategy_description": f"{baseline_desc} {shift_desc}",
                    "baseline_name": baseline_name,
                    "feature_mode": feature_mode,
                    "shift_name": shift_name,
                    "params_file": cfg_rel,
                    "tar_path": meta["tar"],
                    "data_prefix": meta["prefix"],
                    "curr": "0.1",
                    "data_access_mode": "direct_tar",
                    "shared_data_dir": meta["shared"],
                    "task_slug": f"{track_key}_{strategy_id}_n1_s{seed}",
                    "notes": "Focused checkpoint6 follow-up on the unstable drift and masking families.",
                }
            )

    fieldnames = list(rows[0].keys())
    with MATRIX_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    note_lines = [
        f"# HH Track4 Checkpoint6 Drift/Mask Follow-Up Classical ({DATE_LABEL})",
        "",
        f"- matrix csv: `{MATRIX_CSV}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Purpose",
        "- tighten the two unstable checkpoint6 shift families: drift and masking",
        "- characterize how the clean-data and robustness classical winners fail as shift severity grows",
        "",
        "## Baselines",
        *[f"- `{name}`" for name in baselines],
        "",
        "## Shift Ladder",
        *[f"- `{name}`: {desc}" for name, _seed, _cfg, desc in shifts],
    ]
    write_text(NOTES_MD, "\n".join(note_lines) + "\n")
    print(f"Wrote matrix: {MATRIX_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
