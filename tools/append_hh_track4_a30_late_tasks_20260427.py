#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
PYTHON_BIN = "/projects/neuro-collab/conda/neuro-collab-env/bin/python"
LITERATURE_REFRESH = REPO / "tools" / "analyze_hh_track4_a30_literature_surrogate_refresh_20260427.py"
HYBRID_APPEND = REPO / "tools" / "append_hh_track4_a30_hybrid_tasks_20260427.py"
SIMFORMER_PYTHON_BIN = str(REPO / ".venvs" / "simformer_py312_jax0423" / "bin" / "python")
SIMFORMER_NVIDIA_ROOT = REPO / ".venvs" / "simformer_py312_jax0423" / "lib" / "python3.12" / "site-packages" / "nvidia"
SIMFORMER_LD_LIBRARY_PATH = ":".join(
    str(SIMFORMER_NVIDIA_ROOT / part / "lib")
    for part in ("cudnn", "cublas", "cusolver", "cusparse", "cufft", "nccl", "nvjitlink", "cuda_runtime")
)
PYTHONPATH_VALUE = f"{REPO}:/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/vendor/dlkit:${{PYTHONPATH:-}}"
CAMPAIGN_ROOT = IMPORTANT / "optimization_track_20260427_hh_track4_a30_literal_rerun"
RUNS = CAMPAIGN_ROOT / "runs"
PARAMS_TRACK4 = "src/pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml"
SHARED_DATA_DIR = (
    IMPORTANT / "optimization_track_20260411_v100_interactive" / "shared_data" / "track4_hh_full" / "concatenated_data"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append later literature and surrogate tasks to the live A30 manifest.")
    ap.add_argument(
        "--manifest-json",
        default=str(CAMPAIGN_ROOT / "tables" / "hh_track4_a30_literal_rerun_manifest_20260427.json"),
    )
    return ap.parse_args()


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def python_command(script_rel: str, *args: object) -> str:
    cmd = [PYTHON_BIN, str(REPO / script_rel), *map(str, args)]
    return f'cd {shlex.quote(str(REPO))} && env PYTHONPATH="{PYTHONPATH_VALUE}" {shell_join(cmd)}'


def python39_command(script_rel: str, *args: object) -> str:
    cmd = ["python", str(REPO / script_rel), *map(str, args)]
    return f'cd {shlex.quote(str(REPO))} && env PYTHONPATH="{PYTHONPATH_VALUE}" {shell_join(cmd)}'


def simformer_python_command(script_rel: str, *args: object) -> str:
    cmd = [SIMFORMER_PYTHON_BIN, str(REPO / script_rel), *map(str, args)]
    return (
        f'cd {shlex.quote(str(REPO))} && env '
        f'PYTHONPATH="{PYTHONPATH_VALUE}" '
        f'LD_LIBRARY_PATH="{SIMFORMER_LD_LIBRARY_PATH}:${{LD_LIBRARY_PATH:-}}" '
        f'JAX_PLATFORMS="cuda,cpu" '
        f'{shell_join(cmd)}'
    )


def add_task(tasks: list[dict[str, object]], task: dict[str, object]) -> None:
    existing = {str(item["task_id"]) for item in tasks}
    if str(task["task_id"]) in existing:
        return
    tasks.append(task)


def save_dir(*parts: str) -> Path:
    return RUNS.joinpath(*parts)


def campaign_tag(campaign_root: Path) -> str:
    name = campaign_root.name
    return name[len("optimization_track_") :] if name.startswith("optimization_track_") else name


def base_task(
    task_id: str,
    phase: str,
    label: str,
    command: str,
    resource_class: str,
    cpus_per_task: int,
    gpus_per_task: int,
    expected_runtime_sec: float,
    depends_on: list[str] | None = None,
    output_paths: list[str] | None = None,
    notes: str = "",
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "phase": phase,
        "label": label,
        "command": command,
        "resource_class": resource_class,
        "cpus_per_task": cpus_per_task,
        "gpus_per_task": gpus_per_task,
        "expected_runtime_sec": expected_runtime_sec,
        "depends_on": depends_on or [],
        "output_paths": output_paths or [],
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_json)
    manifest = json.loads(manifest_path.read_text())
    tasks: list[dict[str, object]] = list(manifest["tasks"])
    tag = campaign_tag(CAMPAIGN_ROOT)
    common = ["--params", PARAMS_TRACK4, "--data-dir", str(SHARED_DATA_DIR), "--data-prefix", "concatenated_data"]

    def add_py(
        task_id: str,
        phase: str,
        label: str,
        script_rel: str,
        extra: list[object],
        runtime: float,
        gpus: int = 1,
        cpus: int = 12,
        depends: list[str] | None = None,
        output_paths: list[str] | None = None,
    ) -> None:
        run_root = save_dir(phase, label)
        add_task(
            tasks,
            base_task(
                task_id,
                phase,
                label,
                python_command(script_rel, *common, "--save-dir", str(run_root), *extra),
                "gpu2" if gpus == 2 else ("gpu1" if gpus == 1 else "cpu_only"),
                cpus,
                gpus,
                runtime,
                depends_on=depends,
                output_paths=output_paths or [str(run_root / "metrics_summary.json"), str(run_root / "predictions_test.npz")],
                notes="late aligned or surrogate branch appended after initial literal A30 manifest build",
            ),
        )

    # aligned posterior families
    add_py("a30runall__bayesflow_native_meanstd_v100_20260425", "phaseLIT_aligned_runall", "bayesflow_native_meanstd_v100_20260425", "tools/run_hh_track4_aligned_multicurrent_bayesflow_20260425.py", ["--aggregation", "meanstd", "--seed", "20260425"], 240.0)
    add_py("a30runall__bayesflow_native_v100_20260425", "phaseLIT_aligned_runall", "bayesflow_native_v100_20260425", "tools/run_hh_track4_aligned_multicurrent_bayesflow_20260425.py", ["--aggregation", "concat", "--seed", "20260425"], 260.0)
    add_py("a30runall__bayesflow_native_structured_currfusion_v100_20260425", "phaseLIT_aligned_runall", "bayesflow_native_structured_currfusion_v100_20260425", "tools/run_hh_track4_aligned_multicurrent_bayesflow_structured_20260425.py", ["--structured-mode", "currfusion_chunkstats", "--seed", "20260425"], 260.0)
    add_py("a30runall__bayesflow_native_structured_currfusion_deltas_v100_20260425", "phaseLIT_aligned_runall", "bayesflow_native_structured_currfusion_deltas_v100_20260425", "tools/run_hh_track4_aligned_multicurrent_bayesflow_structured_20260425.py", ["--structured-mode", "currfusion_chunkstats_deltas", "--seed", "20260425"], 260.0)
    add_py("a30runall__bayesflow_temporal_v100_20260426", "phaseLIT_aligned_runall", "bayesflow_temporal_v100_20260426", "tools/run_hh_track4_aligned_multicurrent_bayesflow_temporal_20260426.py", ["--summary-network", "time_series_network", "--seed", "20260426"], 300.0)
    add_py("a30runall__bayesflow_temporal_transformer_v100_20260426", "phaseLIT_aligned_runall", "bayesflow_temporal_transformer_v100_20260426", "tools/run_hh_track4_aligned_multicurrent_bayesflow_temporal_20260426.py", ["--summary-network", "time_series_transformer", "--seed", "20260426", "--time-points", "512", "--batch-size", "16", "--summary-dim", "32"], 420.0)
    add_py("a30runall__bayesflow_fusion_transformer_v100_20260426", "phaseLIT_aligned_runall", "bayesflow_fusion_transformer_v100_20260426", "tools/run_hh_track4_aligned_multicurrent_bayesflow_temporal_20260426.py", ["--summary-network", "fusion_transformer", "--seed", "20260426", "--time-points", "512", "--batch-size", "16", "--summary-dim", "32"], 360.0)
    add_py("a30runall__bayesflow_set_posterior_v100_20260426", "phaseLIT_aligned_runall", "bayesflow_set_posterior_v100_20260426", "tools/run_hh_track4_aligned_multicurrent_bayesflow_set_20260426.py", ["--summary-network", "set_transformer", "--seed", "20260426"], 260.0)
    add_py("a30runall__bayesflow_deepset_posterior_v100_20260426", "phaseLIT_aligned_runall", "bayesflow_deepset_posterior_v100_20260426", "tools/run_hh_track4_aligned_multicurrent_bayesflow_set_20260426.py", ["--summary-network", "deep_set", "--seed", "20260426"], 260.0)
    add_py("a30runall__swyft_native_v100_20260425", "phaseLIT_aligned_runall", "swyft_native_v100_20260425", "tools/run_hh_track4_aligned_multicurrent_swyft_20260425.py", ["--aggregation", "meanstd", "--seed", "20260425", "--device", "gpu"], 380.0)
    add_py("a30runall__swyft_native_concat_small_v100_20260425", "phaseLIT_aligned_runall", "swyft_native_concat_small_v100_20260425", "tools/run_hh_track4_aligned_multicurrent_swyft_20260425.py", ["--aggregation", "concat", "--seed", "20260425", "--batch-size", "64", "--hidden-features", "64", "--num-blocks", "2", "--dropout", "0.05", "--disable-pairwise", "--device", "gpu"], 90.0)
    add_py("a30runall__swyft_native_aligned_currents_pool_v100_20260425", "phaseLIT_aligned_runall", "swyft_native_aligned_currents_pool_v100_20260425", "tools/run_hh_track4_aligned_multicurrent_swyft_20260425.py", ["--aggregation", "concat", "--seed", "20260425", "--batch-size", "64", "--model-variant", "aligned_currents_pool", "--structured-hidden-features", "96", "--device", "gpu"], 220.0)
    add_py("a30runall__swyft_tmnre_native_v100_20260425", "phaseLIT_aligned_runall", "swyft_tmnre_native_v100_20260425", "tools/run_hh_track4_aligned_multicurrent_swyft_tmnre_20260425.py", ["--stage1-run-dir", str(save_dir("phaseLIT_aligned_runall", "swyft_native_v100_20260425")), "--aggregation", "meanstd", "--seed", "20260425", "--batch-size", "64", "--max-epochs", "20", "--device", "gpu"], 430.0, depends=["a30runall__swyft_native_v100_20260425"])
    add_py(
        "a30runall__swyft_tmnre_schedule_matrix_v100_20260425",
        "phaseLIT_aligned_runall",
        "swyft_tmnre_schedule_matrix_v100_20260425",
        "tools/run_hh_track4_aligned_multicurrent_swyft_tmnre_schedule_matrix_20260425.py",
        ["--stage1-run-dir", str(save_dir("phaseLIT_aligned_runall", "swyft_native_v100_20260425")), "--aggregation", "meanstd", "--device", "gpu"],
        430.0,
        depends=["a30runall__swyft_native_v100_20260425"],
        output_paths=[
            str(save_dir("phaseLIT_aligned_runall", "swyft_tmnre_schedule_matrix_v100_20260425") / "tmnre_schedule_matrix_summary.csv"),
            str(save_dir("phaseLIT_aligned_runall", "swyft_tmnre_schedule_matrix_v100_20260425") / "tmnre_schedule_matrix_report.md"),
        ],
    )
    add_py("a30runall__swyft_tmnre_structured_stage1_v100_20260425", "phaseLIT_aligned_runall", "swyft_tmnre_structured_stage1_v100_20260425", "tools/run_hh_track4_aligned_multicurrent_swyft_tmnre_20260425.py", ["--stage1-run-dir", str(save_dir("phaseLIT_aligned_runall", "swyft_native_aligned_currents_pool_v100_20260425")), "--aggregation", "concat", "--model-variant", "aligned_currents_pool", "--structured-hidden-features", "96", "--seed", "20260425", "--batch-size", "64", "--max-epochs", "20", "--device", "gpu"], 220.0, depends=["a30runall__swyft_native_aligned_currents_pool_v100_20260425"])
    add_py("a30runall__swyft_tmnre_score_prune_structured_v100_20260426", "phaseLIT_aligned_runall", "swyft_tmnre_score_prune_structured_v100_20260426", "tools/run_hh_track4_aligned_multicurrent_swyft_tmnre_score_prune_20260426.py", ["--stage1-run-dir", str(save_dir("phaseLIT_aligned_runall", "swyft_native_aligned_currents_pool_v100_20260425")), "--aggregation", "concat", "--seed", "20260426", "--device", "gpu"], 80.0, depends=["a30runall__swyft_native_aligned_currents_pool_v100_20260425"])
    add_py("a30runall__temporal_cnn_aligned_v100_20260426", "phaseLIT_aligned_runall", "temporal_cnn_aligned_v100_20260426", "tools/run_hh_track4_aligned_multicurrent_temporal_cnn_20260426.py", ["--seed", "20260426"], 240.0)

    # pool sequential SBI
    for task_id, label, method, density, sample_with, rounds, initial_train_size, runtime in [
        ("a30runall__fmpe_mlp_r2_pool4096_final2048", "fmpe_mlp_r2_pool4096_final2048", "fmpe", "mlp", "ode", "2", "1024", 704.0),
        ("a30runall__snpe_maf_r3_pool4096_final2048", "snpe_maf_r3_pool4096_final2048", "snpe", "maf", "direct", "3", "512", 98.0),
    ]:
        run_root = save_dir("phaseLIT_aligned_runall", label)
        cmd = python_command(
            "tools/run_hh_track4_pool_sequential_sbi_20260425.py",
            "--params", PARAMS_TRACK4,
            "--save-dir", str(run_root),
            "--method", method,
            "--density-estimator", density,
            "--sample-with", sample_with,
            "--pool-size", "4096",
            "--final-train-size", "2048",
            "--n-validate", "1024",
            "--n-test", "1024",
            "--data-dir", str(SHARED_DATA_DIR),
            "--data-prefix", "concatenated_data",
            "--curr", "0.1",
            "--device", "cpu",
            "--feature-mode", "raw_plus_fft256_summary12",
            "--seed", "20260425",
            "--embedding-dim", "64",
            "--embedding-hidden", "256",
            "--hidden-features", "128",
            "--num-transforms", "5",
            "--training-batch-size", "128",
            "--learning-rate", "5.0e-4",
            "--stop-after-epochs", "12",
            "--max-num-epochs", "60",
            "--posterior-samples", "8",
            "--decision-rule", "mean",
            "--rounds", rounds,
            "--initial-train-size", initial_train_size,
            "--candidate-pool-size", "1024",
            "--candidate-posterior-samples", "4",
        )
        add_task(tasks, base_task(task_id, "phaseLIT_aligned_runall", label, cmd, "cpu_only", 8, 0, runtime, output_paths=[str(run_root / "predictions_test.npz")], notes="pool sequential SBI branch"))

    # surrogate active and exact ASNPE
    for task_id, label, extra, runtime in [
        ("a30surr__active_sequential_smoke_20260425", "active_sequential_smoke_20260425", ["--n-episodes", "1", "--n-particles", "64", "--n-rounds", "2"], 8.0),
        ("a30surr__active_sequential_wasserstein_smoke_20260425", "active_sequential_wasserstein_smoke_20260425", ["--acquisition-policy", "wasserstein", "--n-episodes", "1", "--n-particles", "64", "--n-rounds", "2"], 45.0),
        ("a30surr__active_sequential_default_20260425", "active_sequential_default_20260425", ["--n-episodes", "4", "--n-particles", "128", "--n-rounds", "4"], 35.0),
        ("a30surr__active_sequential_disagreement_full_20260425", "active_sequential_disagreement_full_20260425", ["--acquisition-policy", "disagreement", "--n-episodes", "4", "--n-particles", "128", "--n-rounds", "4"], 40.0),
        ("a30surr__active_sequential_wasserstein_full_20260425", "active_sequential_wasserstein_full_20260425", ["--acquisition-policy", "wasserstein", "--n-episodes", "4", "--n-particles", "128", "--n-rounds", "4"], 40.0),
        ("a30surr__active_sequential_uncertainty_full_20260425", "active_sequential_uncertainty_full_20260425", ["--acquisition-policy", "uncertainty", "--n-episodes", "4", "--n-particles", "128", "--n-rounds", "4"], 40.0),
    ]:
        run_root = save_dir("phaseSURR_active", label)
        cmd = python_command("tools/run_hh_track4_assumption_conditioned_active_sequential_design_20260425.py", "--save-dir", str(run_root), *extra)
        add_task(tasks, base_task(task_id, "phaseSURR_active", label, cmd, "cpu_only", 8, 0, runtime, output_paths=[str(run_root / "active_sequential_manifest.json")], notes="surrogate active-design branch"))

    for task_id, label, extra, runtime in [
        ("a30surr__asnpe_primary_v100_20260426", "asnpe_primary_v100_20260426", ["--save-dir", str(save_dir("phaseSURR_active", "asnpe_primary_v100_20260426")), "--bundle-label", "primary_a30", "--n-rounds", "5", "--ensemble-size", "4", "--epochs", "24", "--batch-size", "32", "--posterior-samples", "1024", "--candidate-pool", "96", "--initial-sims", "128", "--round-sims", "32"], 20.0),
        ("a30surr__asnpe_smoke_local_20260426", "asnpe_smoke_local_20260426", ["--save-dir", str(save_dir("phaseSURR_active", "asnpe_smoke_local_20260426")), "--bundle-label", "smoke_local_a30", "--n-rounds", "3", "--ensemble-size", "2", "--epochs", "12", "--batch-size", "64", "--posterior-samples", "256", "--candidate-pool", "64", "--initial-sims", "64", "--round-sims", "16"], 12.0),
        ("a30surr__asnpe_secondary_v100_20260426", "asnpe_secondary_v100_20260426", ["--save-dir", str(save_dir("phaseSURR_active", "asnpe_secondary_v100_20260426")), "--bundle-label", "secondary_a30", "--n-rounds", "4", "--ensemble-size", "3", "--epochs", "18", "--batch-size", "48", "--posterior-samples", "512", "--candidate-pool", "96", "--initial-sims", "96", "--round-sims", "24"], 20.0),
    ]:
        cmd = python_command("tools/run_hh_track4_assumption_conditioned_asnpe_20260426.py", *extra)
        run_root = Path(extra[1]) if extra[0] == "--save-dir" else save_dir("phaseSURR_active", label)
        add_task(tasks, base_task(task_id, "phaseSURR_active", label, cmd, "cpu_only", 8, 0, runtime, output_paths=[str(run_root / "asnpe_manifest.json")], notes="exact-source ASNPE-style branch"))

    swyft_sbc_root = save_dir("phaseSURR_aux", "swyft_surrogate_sbc_20260426")
    swyft_stage1 = save_dir("phaseLIT_aligned_runall", "swyft_native_aligned_currents_pool_v100_20260425")
    add_task(
        tasks,
        base_task(
            "a30surr__swyft_surrogate_sbc_20260426",
            "phaseSURR_aux",
            "swyft_surrogate_sbc_20260426",
            python_command(
                "tools/run_hh_track4_swyft_trained_posterior_sbc_20260426.py",
                "--params", PARAMS_TRACK4,
                "--save-dir", str(swyft_sbc_root),
                "--stage1-run-dir", str(swyft_stage1),
                "--aggregation", "concat",
                "--device", "gpu",
            ),
            "gpu1",
            12,
            1,
            180.0,
            depends_on=["a30runall__swyft_native_aligned_currents_pool_v100_20260425"],
            output_paths=[str(swyft_sbc_root / "swyft_surrogate_sbc_manifest.json")],
            notes="trained posterior SBC on the rerun aligned Swyft stage1 branch",
        ),
    )

    # new literature
    gen_root = save_dir("phaseLIT_new_methods", "generalized_bayes_npe_main_20260426")
    add_task(tasks, base_task("a30lit__generalized_bayes_npe_main_20260426", "phaseLIT_new_methods", "generalized_bayes_npe_main_20260426", python_command("tools/run_hh_track4_assumption_conditioned_generalized_bayes_npe_20260426.py", "--save-dir", str(gen_root), "--seed", "20260426"), "cpu_only", 8, 0, 80.0, output_paths=[str(gen_root / "generalized_bayes_npe_manifest.json")], notes="bounded generalized-Bayes NPE adaptation"))
    for task_id, label, runtime in [
        ("a30lit__simformer_official_hh_primary_20260426", "simformer_official_hh_primary_20260426", 40.0),
        ("a30lit__simformer_official_hh_secondary_20260426", "simformer_official_hh_secondary_20260426", 40.0),
    ]:
        run_root = save_dir("phaseLIT_new_methods", label)
        add_task(tasks, base_task(task_id, "phaseLIT_new_methods", label, simformer_python_command("tools/run_hh_track4_simformer_official_hh_20260426.py", "--save-dir", str(run_root)), "gpu1", 12, 1, runtime, output_paths=[str(run_root / "simformer_hh_metrics_summary.json")], notes="official upstream Simformer HH benchmark branch"))

    refresh_depends = [
        str(task["task_id"])
        for task in tasks
        if str(task.get("phase", "")).startswith("phaseLIT_") or str(task.get("phase", "")).startswith("phaseSURR_")
    ]
    tables_dir = CAMPAIGN_ROOT / "tables"
    reports_dir = CAMPAIGN_ROOT / "reports"
    refresh_outputs = [
        tables_dir / f"hh_track4_a30_literature_framework_snapshot_20260427_{tag}.csv",
        tables_dir / f"hh_track4_a30_literature_posterior_decision_rules_20260427_{tag}.csv",
        tables_dir / f"hh_track4_a30_literature_active_policy_comparison_20260427_{tag}.csv",
        tables_dir / f"hh_track4_a30_literature_method_comparison_20260427_{tag}.csv",
        tables_dir / f"hh_track4_a30_literature_branch_registry_20260427_{tag}.csv",
        tables_dir / f"hh_track4_a30_literature_reporting_key_results_20260427_{tag}.csv",
        reports_dir / f"hh_track4_a30_literature_branch_registry_20260427_{tag}.md",
        reports_dir / f"hh_track4_a30_literature_reporting_synthesis_20260427_{tag}.md",
    ]
    add_task(
        tasks,
        base_task(
            "a30analysis__literature_surrogate_refresh",
            "phaseANALYSIS_literature_surrogate",
            "literature_surrogate_refresh",
            python_command("tools/analyze_hh_track4_a30_literature_surrogate_refresh_20260427.py", "--campaign-root", str(CAMPAIGN_ROOT)),
            "cpu_only",
            4,
            0,
            60.0,
            depends_on=sorted(refresh_depends),
            output_paths=[str(path) for path in refresh_outputs],
            notes="refresh A30-local aligned literature, decision-rule, active-design, and reporting tables after the rerun branches complete",
        ),
    )

    hybrid_summary = CAMPAIGN_ROOT / "notes" / "hh_track4_a30_hybrid_append_summary_20260427.json"
    add_task(
        tasks,
        base_task(
            "a30manifest__append_hybrid_tasks",
            "phaseMANIFEST_hybrid_append",
            "append_hybrid_tasks",
            python_command(
                "tools/append_hh_track4_a30_hybrid_tasks_20260427.py",
                "--campaign-root",
                str(CAMPAIGN_ROOT),
                "--manifest-json",
                str(manifest_path),
            ),
            "cpu_only",
            4,
            0,
            30.0,
            depends_on=["a30analysis__literature_surrogate_refresh"],
            output_paths=[str(hybrid_summary)],
            notes="append the A30-local surrogate direct-fitting hybrid tail after decision-rule refresh completes",
        ),
    )

    manifest["tasks"] = tasks
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"task_count": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
