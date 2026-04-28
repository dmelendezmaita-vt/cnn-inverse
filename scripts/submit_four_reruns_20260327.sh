#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-submit}"
if [[ "${MODE}" != "submit" && "${MODE}" != "plan" ]]; then
  echo "usage: $0 [submit|plan]" >&2
  exit 2
fi

DATE_TAG="${DATE_TAG:-20260327}"
REPO="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch"
TRAIN_SCRIPT="${REPO}/slurm_train.sbatch"

OUT_ROOT="${REPO}/data/important_notes/advisor_packet_${DATE_TAG}/reruns"
TABLES_DIR="${OUT_ROOT}/tables"
NOTES_DIR="${OUT_ROOT}/notes"
LOGS_DIR="${OUT_ROOT}/logs"
CONFIG_ROOT="${REPO}/pytorch/configs/reruns_${DATE_TAG}"

PLAN_CSV="${TABLES_DIR}/rerun_plan_${DATE_TAG}.csv"
JOBS_CSV="${TABLES_DIR}/rerun_jobs_registry_${DATE_TAG}.csv"
PLAN_MD="${NOTES_DIR}/rerun_execution_plan_${DATE_TAG}.md"

mkdir -p "${TABLES_DIR}" "${NOTES_DIR}" "${LOGS_DIR}" "${CONFIG_ROOT}"

python3 - "${REPO}" "${CONFIG_ROOT}" "${PLAN_CSV}" "${PLAN_MD}" <<'PY'
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

repo = Path(sys.argv[1])
config_root = Path(sys.argv[2])
plan_csv = Path(sys.argv[3])
plan_md = Path(sys.argv[4])
date_label = "2026-03-27"


def read_text(path: Path) -> str:
    return path.read_text()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def set_key(text: str, key: str, value: str) -> str:
    pattern = rf"(^\s*{re.escape(key)}:\s*).*$"
    repl = rf"\g<1>{value}"
    new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n == 0:
        raise ValueError(f"Key '{key}' not found while patching config.")
    return new_text


def fmt_lr(v: float) -> str:
    s = f"{v:.1e}"
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return s


def slug_float(v: float) -> str:
    s = fmt_lr(v)
    s = s.replace(".", "p").replace("+", "").replace("-", "m")
    return s


def create_config(
    base_rel: str,
    out_rel: str,
    updates: dict[str, str],
) -> str:
    base_path = repo / base_rel
    out_path = repo / out_rel
    txt = read_text(base_path)
    for k, v in updates.items():
        txt = set_key(txt, k, v)
    write_text(out_path, txt)
    return out_rel


rows: list[dict[str, str]] = []


def add_row(
    *,
    rerun_group: str,
    track_key: str,
    sweep_variable: str,
    sweep_value: str,
    job_name: str,
    params_file: str,
    tar_path: str,
    data_prefix: str,
    nodes: int,
    gpus_per_node: int,
    cpus_per_node: int,
    mem_gib: int,
    partition: str = "a30_normal_q",
    qos: str = "fal_a30_normal_short",
    time_limit: str = "01:30:00",
    gpu_token: str = "a30",
    gpu: str = "A30",
    cluster: str = "Falcon",
    data_access_mode: str = "copy_to_node",
    notes: str = "",
) -> None:
    rows.append(
        {
            "rerun_group": rerun_group,
            "track_key": track_key,
            "sweep_variable": sweep_variable,
            "sweep_value": sweep_value,
            "job_name": job_name,
            "params_file": params_file,
            "tar_path": tar_path,
            "data_prefix": data_prefix,
            "nodes": str(nodes),
            "gpus_per_node": str(gpus_per_node),
            "cpus_per_node": str(cpus_per_node),
            "mem_gib": str(mem_gib),
            "partition": partition,
            "qos": qos,
            "time_limit": time_limit,
            "gpu_token": gpu_token,
            "gpu": gpu,
            "cluster": cluster,
            "data_access_mode": data_access_mode,
            "notes": notes,
        }
    )


# 1) True weak scaling rerun (Track 2, A30 only)
track2_base = "pytorch/configs/falcon_seed_repeats/params_dnn_tar_seed301.yaml"
track2_tar = "/projects/neuro-collab/data/tar_files/fhn_publication_2020.tar"
track2_prefix = "publication_2020"
for nodes, ntrain, gbs in [(1, 1000, 32), (2, 2000, 64), (4, 4000, 128)]:
    cfg_rel = f"pytorch/configs/reruns_20260327/track2_true_weak/params_t2_true_weak_a30_{nodes}n.yaml"
    create_config(
        track2_base,
        cfg_rel,
        {
            "description": f"\"Track 2 true weak scaling rerun ({nodes} node A30) {date_label}\"",
            "Ntrain": str(ntrain),
            "global_train_batch_size": str(gbs),
            "train_batch_size": str(gbs),
            "random_seed": "401",
        },
    )
    add_row(
        rerun_group="true_weak_scaling_track2",
        track_key="track2_fhn_scalability",
        sweep_variable="nodes_ntrain_global_batch",
        sweep_value=f"nodes={nodes}|Ntrain={ntrain}|global_batch={gbs}",
        job_name=f"trn_rerun_t2weak_A30_{nodes}n",
        params_file=cfg_rel,
        tar_path=track2_tar,
        data_prefix=track2_prefix,
        nodes=nodes,
        gpus_per_node=4,
        cpus_per_node=48,
        mem_gib=64,
        notes="true weak scaling style sweep on A30 only",
    )


# Shared metadata for Track 3 and Track 4 reruns
track_meta = {
    "track3_hh_reduced": {
        "base": "pytorch/configs/third_track_hh/params_dnn_tar_hh.yaml",
        "tar": "/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
        "prefix": "reduced_data",
        "mem": 128,
    },
    "track4_hh_full": {
        "base": "pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml",
        "tar": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
        "prefix": "concatenated_data",
        "mem": 192,
    },
}


# 2) LR/epoch sweep
lr_epoch_combos = [(5.0e-3, 200), (1.0e-2, 200), (1.0e-2, 300)]
for track_key, meta in track_meta.items():
    for lr, ep in lr_epoch_combos:
        lr_slug = slug_float(lr)
        cfg_rel = (
            f"pytorch/configs/reruns_20260327/{track_key}/"
            f"params_{track_key}_lr{lr_slug}_ep{ep}.yaml"
        )
        create_config(
            meta["base"],
            cfg_rel,
            {
                "description": f"\"{track_key} lr-epoch sweep rerun {date_label}\"",
                "learning_rate": fmt_lr(lr),
                "epochs": str(ep),
                "random_seed": "423",
            },
        )
        add_row(
            rerun_group="lr_epoch_sweep_track3_track4",
            track_key=track_key,
            sweep_variable="learning_rate_epochs",
            sweep_value=f"learning_rate={fmt_lr(lr)}|epochs={ep}",
            job_name=f"trn_rerun_{track_key}_lr{lr_slug}_ep{ep}",
            params_file=cfg_rel,
            tar_path=meta["tar"],
            data_prefix=meta["prefix"],
            nodes=1,
            gpus_per_node=4,
            cpus_per_node=48,
            mem_gib=int(meta["mem"]),
            notes="controlled lr and epoch sweep",
        )


# 3) Batch-size sweep
for track_key, meta in track_meta.items():
    for gbs in [32, 64, 128]:
        cfg_rel = (
            f"pytorch/configs/reruns_20260327/{track_key}/"
            f"params_{track_key}_gbs{gbs}.yaml"
        )
        create_config(
            meta["base"],
            cfg_rel,
            {
                "description": f"\"{track_key} global batch sweep rerun {date_label}\"",
                "global_train_batch_size": str(gbs),
                "train_batch_size": str(gbs),
                "random_seed": "424",
            },
        )
        add_row(
            rerun_group="batch_size_sweep_track3_track4",
            track_key=track_key,
            sweep_variable="global_train_batch_size",
            sweep_value=f"global_train_batch_size={gbs}",
            job_name=f"trn_rerun_{track_key}_gbs{gbs}",
            params_file=cfg_rel,
            tar_path=meta["tar"],
            data_prefix=meta["prefix"],
            nodes=1,
            gpus_per_node=4,
            cpus_per_node=48,
            mem_gib=int(meta["mem"]),
            notes="expected training accuracy trend check",
        )


# 4) Input-length ablation
for track_key, meta in track_meta.items():
    for sub_len in [500, 1000, 2000]:
        cfg_rel = (
            f"pytorch/configs/reruns_20260327/{track_key}/"
            f"params_{track_key}_sub{sub_len}.yaml"
        )
        create_config(
            meta["base"],
            cfg_rel,
            {
                "description": f"\"{track_key} input length ablation rerun {date_label}\"",
                "features_sub_length": str(sub_len),
                "random_seed": "425",
            },
        )
        add_row(
            rerun_group="input_length_ablation_track3_track4",
            track_key=track_key,
            sweep_variable="features_sub_length",
            sweep_value=f"features_sub_length={sub_len}",
            job_name=f"trn_rerun_{track_key}_sub{sub_len}",
            params_file=cfg_rel,
            tar_path=meta["tar"],
            data_prefix=meta["prefix"],
            nodes=1,
            gpus_per_node=4,
            cpus_per_node=48,
            mem_gib=int(meta["mem"]),
            notes="vector size sensitivity test",
        )


fieldnames = [
    "rerun_group",
    "track_key",
    "sweep_variable",
    "sweep_value",
    "job_name",
    "params_file",
    "tar_path",
    "data_prefix",
    "nodes",
    "gpus_per_node",
    "cpus_per_node",
    "mem_gib",
    "partition",
    "qos",
    "time_limit",
    "gpu_token",
    "gpu",
    "cluster",
    "data_access_mode",
    "notes",
]
plan_csv.parent.mkdir(parents=True, exist_ok=True)
with plan_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow(row)

counts: dict[str, int] = {}
for row in rows:
    counts[row["rerun_group"]] = counts.get(row["rerun_group"], 0) + 1

lines = [
    f"# Four Reruns Execution Plan ({date_label})",
    "",
    "Training-only jobs (testing runs excluded).",
    "",
    f"Total jobs planned: {len(rows)}",
]
for k in sorted(counts):
    lines.append(f"- {k}: {counts[k]} jobs")

lines.extend(
    [
        "",
        "Groups:",
        "- true_weak_scaling_track2: Track 2 A30 weak-scaling style rerun (1/2/4 nodes).",
        "- lr_epoch_sweep_track3_track4: learning-rate and epoch sweep on Track 3 and Track 4.",
        "- batch_size_sweep_track3_track4: global batch size sweep on Track 3 and Track 4.",
        "- input_length_ablation_track3_track4: input sub-length ablation on Track 3 and Track 4.",
    ]
)
plan_md.parent.mkdir(parents=True, exist_ok=True)
plan_md.write_text("\n".join(lines) + "\n")
PY

run_falcon_bash() {
  local cmd="$1"
  local quoted
  printf -v quoted "%q" "${cmd}"
  ssh -n -o BatchMode=yes tinkercliffs1 ssh -n -o BatchMode=yes falcon1.arc.vt.edu bash -lc "${quoted}"
}

if [[ "${MODE}" == "plan" ]]; then
  cat "${PLAN_MD}"
  echo
  echo "Plan CSV: ${PLAN_CSV}"
  exit 0
fi

timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
{
  echo "rerun_group,track_key,sweep_variable,sweep_value,job_name,job_id,status,elapsed,cluster,gpu,nodes,gpus_per_node,cpus_per_node,mem_gib,partition,qos,time_limit,params_file,tar_path,data_prefix,data_access_mode,submitted_at,submit_action,notes"
  while IFS=, read -r rerun_group track_key sweep_variable sweep_value job_name params_file tar_path data_prefix nodes gpus_per_node cpus_per_node mem_gib partition qos time_limit gpu_token gpu cluster data_access_mode notes; do
    [[ "${rerun_group}" == "rerun_group" ]] && continue

    rerun_group="${rerun_group//$'\r'/}"
    track_key="${track_key//$'\r'/}"
    sweep_variable="${sweep_variable//$'\r'/}"
    sweep_value="${sweep_value//$'\r'/}"
    job_name="${job_name//$'\r'/}"
    params_file="${params_file//$'\r'/}"
    tar_path="${tar_path//$'\r'/}"
    data_prefix="${data_prefix//$'\r'/}"
    nodes="${nodes//$'\r'/}"
    gpus_per_node="${gpus_per_node//$'\r'/}"
    cpus_per_node="${cpus_per_node//$'\r'/}"
    mem_gib="${mem_gib//$'\r'/}"
    partition="${partition//$'\r'/}"
    qos="${qos//$'\r'/}"
    time_limit="${time_limit//$'\r'/}"
    gpu_token="${gpu_token//$'\r'/}"
    gpu="${gpu//$'\r'/}"
    cluster="${cluster//$'\r'/}"
    data_access_mode="${data_access_mode//$'\r'/}"
    notes="${notes//$'\r'/}"

    export_block="ALL,DATA_ACCESS_MODE=${data_access_mode},TAR_PATH=${tar_path},DATA_PREFIX=${data_prefix},CURR=0.1,PARAMS_FILE=${params_file},TORCH_NCCL_ASYNC_ERROR_HANDLING=1,TORCH_NCCL_BLOCKING_WAIT=1,NCCL_DEBUG=WARN"
    cmd="cd ${REPO} && sbatch --parsable --job-name=${job_name} --account=neuro-collab --partition=${partition} --qos=${qos} --nodes=${nodes} --ntasks-per-node=1 --cpus-per-task=${cpus_per_node} --mem=${mem_gib}G --gres=gpu:${gpu_token}:${gpus_per_node} --time=${time_limit} --export=${export_block} ${TRAIN_SCRIPT}"

    job_id="$(run_falcon_bash "${cmd}")"
    job_id="${job_id%%;*}"
    status="SUBMITTED"
    elapsed="0:00"
    submit_action="submitted_now"

    echo "${rerun_group},${track_key},${sweep_variable},${sweep_value},${job_name},${job_id},${status},${elapsed},${cluster},${gpu},${nodes},${gpus_per_node},${cpus_per_node},${mem_gib},${partition},${qos},${time_limit},${params_file},${tar_path},${data_prefix},${data_access_mode},${timestamp},${submit_action},${notes}"
  done < "${PLAN_CSV}"
} > "${JOBS_CSV}"

echo "Submitted/recorded rerun jobs:"
echo "  Plan: ${PLAN_CSV}"
echo "  Jobs: ${JOBS_CSV}"
echo "  Notes: ${PLAN_MD}"
