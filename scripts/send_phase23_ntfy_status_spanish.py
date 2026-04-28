#!/usr/bin/env python3
"""Send periodic ntfy updates in Spanish about Phase 2+3 Slurm jobs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PENDING_STATES = {
    "PENDING",
    "CONFIGURING",
    "SUSPENDED",
    "RESV_DEL_HOLD",
    "REQUEUE_HOLD",
}
RUNNING_STATES = {
    "RUNNING",
    "COMPLETING",
    "STAGE_OUT",
    "SIGNALING",
}
COMPLETED_STATES = {"COMPLETED"}
FAILED_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def run_cmd(args: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def load_jobs(registry_csv: Path) -> Tuple[List[str], Dict[str, str]]:
    job_ids: List[str] = []
    phase_by_job: Dict[str, str] = {}
    with registry_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jid = (row.get("job_id") or "").strip()
            if not jid.isdigit():
                continue
            phase = (row.get("phase") or "unknown").strip() or "unknown"
            job_ids.append(jid)
            phase_by_job[jid] = phase
    unique_ids = sorted(set(job_ids), key=int)
    if not unique_ids:
        raise RuntimeError(f"No valid job IDs found in {registry_csv}")
    return unique_ids, phase_by_job


def query_squeue(job_ids: List[str], cluster: str = "Falcon") -> Dict[str, Dict[str, str]]:
    ids_arg = ",".join(job_ids)
    cmd = ["squeue", "-h"]
    if cluster:
        cmd.extend(["-M", cluster])
    cmd.extend(["-j", ids_arg, "-o", "%A|%T|%M|%R"])
    rc, out, _ = run_cmd(cmd)
    rows: Dict[str, Dict[str, str]] = {}
    if rc != 0 or not out:
        return rows
    for line in out.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        jid, state, elapsed, reason = [p.strip() for p in parts]
        if not jid.isdigit():
            continue
        rows[jid] = {
            "state": state.upper(),
            "elapsed": elapsed,
            "reason": reason,
            "source": "squeue",
        }
    return rows


def normalize_state(raw_state: str) -> str:
    state = raw_state.strip().upper()
    if not state:
        return "UNKNOWN"
    # sacct sometimes returns values like "CANCELLED by 12345"
    if " " in state:
        state = state.split(" ", 1)[0]
    return state


def query_sacct(job_ids: List[str], cluster: str = "Falcon") -> Dict[str, Dict[str, str]]:
    ids_arg = ",".join(job_ids)
    cmd = ["sacct", "-n", "-P", "-X"]
    if cluster:
        cmd.extend(["-M", cluster])
    cmd.extend(["-j", ids_arg, "--format=JobIDRaw,State,Elapsed"])
    rc, out, _ = run_cmd(cmd)
    rows: Dict[str, Dict[str, str]] = {}
    if rc != 0 or not out:
        return rows
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        jid, state, elapsed = [p.strip() for p in parts]
        if not jid.isdigit():
            continue
        rows[jid] = {
            "state": normalize_state(state),
            "elapsed": elapsed,
            "reason": "",
            "source": "sacct",
        }
    return rows


def classify_state(state: str) -> str:
    st = normalize_state(state)
    if st in PENDING_STATES:
        return "pendientes"
    if st in RUNNING_STATES:
        return "ejecutando"
    if st in COMPLETED_STATES:
        return "completados"
    if st in FAILED_STATES or st.startswith("CANCELLED"):
        return "fallidos"
    if st in {"UNKNOWN", ""}:
        return "desconocidos"
    return "otros"


def collect_status(job_ids: List[str], cluster: str = "Falcon") -> Dict[str, Dict[str, str]]:
    sq = query_squeue(job_ids, cluster=cluster)
    sa = query_sacct(job_ids, cluster=cluster)
    combined: Dict[str, Dict[str, str]] = {}
    for jid in job_ids:
        if jid in sq:
            combined[jid] = sq[jid]
        elif jid in sa:
            combined[jid] = sa[jid]
        else:
            combined[jid] = {
                "state": "UNKNOWN",
                "elapsed": "",
                "reason": "",
                "source": "none",
            }
    return combined


def build_message(
    iteration: int,
    total_iterations_hint: str,
    status_by_job: Dict[str, Dict[str, str]],
    phase_by_job: Dict[str, str],
    next_sleep_s: int,
) -> str:
    now = dt.datetime.now().astimezone()
    groups = Counter()
    running_ids: List[str] = []
    pending_ids: List[str] = []
    failed_ids: List[str] = []

    phase_groups: Dict[str, Counter] = defaultdict(Counter)

    for jid, meta in status_by_job.items():
        group = classify_state(meta.get("state", "UNKNOWN"))
        groups[group] += 1
        phase = phase_by_job.get(jid, "unknown")
        phase_groups[phase][group] += 1

        if group == "ejecutando":
            running_ids.append(jid)
        elif group == "pendientes":
            pending_ids.append(jid)
        elif group == "fallidos":
            failed_ids.append(jid)

    lines = [
        f"Actualizacion Falcon fase2+fase3 ({iteration}/{total_iterations_hint})",
        f"Hora: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Total de jobs monitoreados: {len(status_by_job)}",
        f"Pendientes: {groups['pendientes']}",
        f"Ejecutando: {groups['ejecutando']}",
        f"Completados: {groups['completados']}",
        f"Fallidos/Cancelados/Timeout: {groups['fallidos']}",
        f"Otros: {groups['otros']}",
        f"Desconocidos (sin estado): {groups['desconocidos']}",
    ]

    for phase in sorted(phase_groups):
        c = phase_groups[phase]
        lines.append(
            "{} -> P:{} R:{} C:{} F:{} O:{} D:{}".format(
                phase,
                c["pendientes"],
                c["ejecutando"],
                c["completados"],
                c["fallidos"],
                c["otros"],
                c["desconocidos"],
            )
        )

    if running_ids:
        lines.append("Ejemplo jobs ejecutando: " + ", ".join(running_ids[:8]))
    elif pending_ids:
        lines.append("Ejemplo jobs pendientes: " + ", ".join(pending_ids[:8]))

    if failed_ids:
        lines.append("Atencion: hay jobs con error. Ejemplos: " + ", ".join(failed_ids[:8]))

    if next_sleep_s > 0:
        lines.append(f"Proxima notificacion en aprox. {next_sleep_s} segundos")
    else:
        lines.append("Fin de ventana de monitoreo")

    return "\n".join(lines)


def send_ntfy(topic: str, title: str, body: str, timeout_s: int = 20) -> Tuple[bool, str]:
    req = urllib.request.Request(
        url=f"https://ntfy.sh/{topic}",
        data=body.encode("utf-8"),
        method="POST",
        headers={
            "Title": title,
            "Priority": "default",
            "Tags": "computer,slurm",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8", errors="replace").strip()
            return True, f"http={resp.status} resp={payload}"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        return False, f"http={exc.code} resp={detail}"
    except Exception as exc:  # noqa: BLE001
        return False, f"error={exc}"


def append_log(log_path: Path, row: Dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-csv", required=True, help="CSV with submitted phase2+phase3 jobs")
    parser.add_argument("--topic", default="dmelendezmaita", help="ntfy topic")
    parser.add_argument("--duration-min", type=int, default=25, help="Total monitor duration in minutes")
    parser.add_argument("--min-interval-sec", type=int, default=180, help="Minimum interval between messages")
    parser.add_argument("--max-interval-sec", type=int, default=240, help="Maximum interval between messages")
    parser.add_argument("--cluster", default="Falcon", help="Slurm cluster name (for squeue/sacct -M)")
    parser.add_argument("--log", required=True, help="Path to jsonl log")
    args = parser.parse_args()

    if args.min_interval_sec <= 0 or args.max_interval_sec <= 0:
        raise SystemExit("Intervals must be positive")
    if args.min_interval_sec > args.max_interval_sec:
        raise SystemExit("min-interval-sec cannot exceed max-interval-sec")

    registry_csv = Path(args.registry_csv)
    log_path = Path(args.log)

    job_ids, phase_by_job = load_jobs(registry_csv)
    started_at = dt.datetime.now().astimezone()
    duration_s = int(args.duration_min * 60)
    end_ts = time.time() + duration_s

    hints = max(1, duration_s // max(1, (args.min_interval_sec + args.max_interval_sec) // 2) + 1)
    total_iterations_hint = str(hints)

    iteration = 0
    while True:
        now_ts = time.time()
        if now_ts > end_ts:
            break

        iteration += 1
        remaining = int(max(0, end_ts - now_ts))
        if remaining == 0:
            next_sleep = 0
        else:
            next_sleep = random.randint(args.min_interval_sec, args.max_interval_sec)
            next_sleep = min(next_sleep, remaining)

        status_by_job = collect_status(job_ids, cluster=args.cluster)
        message = build_message(
            iteration=iteration,
            total_iterations_hint=total_iterations_hint,
            status_by_job=status_by_job,
            phase_by_job=phase_by_job,
            next_sleep_s=next_sleep,
        )
        title = f"Estado jobs Falcon {iteration}"
        ok, resp = send_ntfy(args.topic, title, message)

        event = {
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "iteration": iteration,
            "title": title,
            "ok": ok,
            "response": resp,
            "next_sleep_s": next_sleep,
        }
        append_log(log_path, event)
        print(json.dumps(event, ensure_ascii=True), flush=True)

        if next_sleep <= 0:
            break
        time.sleep(next_sleep)

    # Final explicit end-of-window notification.
    status_by_job = collect_status(job_ids, cluster=args.cluster)
    final_message = build_message(
        iteration=iteration + 1,
        total_iterations_hint=total_iterations_hint,
        status_by_job=status_by_job,
        phase_by_job=phase_by_job,
        next_sleep_s=0,
    )
    final_message = (
        final_message
        + "\nResumen: termino el monitoreo automatico de 25 minutos solicitado."
        + f"\nInicio: {started_at.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        + f"\nFin: {dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    ok, resp = send_ntfy(args.topic, "Estado jobs Falcon (fin de monitoreo)", final_message)
    end_event = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "iteration": iteration + 1,
        "title": "Estado jobs Falcon (fin de monitoreo)",
        "ok": ok,
        "response": resp,
        "next_sleep_s": 0,
    }
    append_log(log_path, end_event)
    print(json.dumps(end_event, ensure_ascii=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
