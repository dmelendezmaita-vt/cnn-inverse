#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shlex
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO = Path('/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch')
RUNS_ROOT = Path('/projects/neuro-collab/data/runs')
OUT_DIR = REPO / 'data/important_notes/publishability_analysis_20260309'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRACK_FILES = {
    'track1_first_parity': [
        REPO / 'data/important_notes/first_track_paper_parity/v100_paper_data_parity_training_jobs.csv',
    ],
    'track2_scalability': [
        REPO / 'data/important_notes/second_track_scalability/training_jobs.csv',
        REPO / 'data/important_notes/second_track_scalability/testing_jobs.csv',
        REPO / 'data/important_notes/second_track_scalability/falcon_seeded_training_jobs.csv',
        REPO / 'data/important_notes/second_track_scalability/tinkercliffs_seeded_training_jobs.csv',
    ],
    'track3_hh_reduced': [
        REPO / 'data/important_notes/third_track_hh/hh_training_jobs.csv',
        REPO / 'data/important_notes/third_track_hh/hh_testing_jobs.csv',
        REPO / 'data/important_notes/third_track_hh/hh_seeded_training_jobs.csv',
        REPO / 'data/important_notes/third_track_hh/hh_seeded_testing_jobs.csv',
    ],
    'track4_hh_full': [
        REPO / 'data/important_notes/fourth_track_hh_full/hh_full_training_jobs.csv',
        REPO / 'data/important_notes/fourth_track_hh_full/hh_full_testing_jobs.csv',
        REPO / 'data/important_notes/fourth_track_hh_full/hh_full_seeded_training_jobs.csv',
        REPO / 'data/important_notes/fourth_track_hh_full/hh_full_seeded_testing_jobs.csv',
    ],
}

REQUIRED_STRICT = [
    'params.yaml',
    'split_indices.npz',
    'net.txt',
    'metrics_summary.json',
    'metrics_summary.csv',
    'predictions.npz',
]

FAILED_LIKE = {'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED', 'PREEMPTED', 'NODE_FAIL'}


@dataclass
class Row:
    track: str
    csv_source: str
    cluster: str
    gpu: str
    node_type: str
    job_name: str
    job_id: str
    qos: str
    nodes: str
    gpus_per_node: str
    cpus_per_node: str
    ram_per_node_gib: str
    status_csv: str
    elapsed_csv: str
    timelimit_csv: str
    data_access_mode: str


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True)


def run_tc(shell_cmd: str) -> subprocess.CompletedProcess:
    remote = f"bash -lc {shlex.quote(shell_cmd)}"
    return run(['ssh', '-n', '-o', 'BatchMode=yes', 'tinkercliffs1', remote])


def run_fal(shell_cmd: str) -> subprocess.CompletedProcess:
    inner = f"bash -lc {shlex.quote(shell_cmd)}"
    outer = f"ssh -o BatchMode=yes falcon1.arc.vt.edu {shlex.quote(inner)}"
    return run(['ssh', '-n', '-o', 'BatchMode=yes', 'tinkercliffs1', outer])


def norm_state(raw: str) -> str:
    s = (raw or '').strip().upper()
    if not s:
        return ''
    t = s.split()[0]
    if t.startswith('COMPLETED'):
        return 'COMPLETED'
    if t in ('RUNNING', 'R'):
        return 'RUNNING'
    if t in ('PENDING', 'PD', 'CF', 'CONFIGURING', 'SUBMITTED'):
        return 'PENDING'
    if t.startswith('COMPLETING'):
        return 'RUNNING'
    if t.startswith('FAILED'):
        return 'FAILED'
    if t.startswith('TIMEOUT'):
        return 'TIMEOUT'
    if t.startswith('CANCELLED'):
        return 'CANCELLED'
    if t.startswith('OUT_OF_MEMORY') or t == 'OOM':
        return 'OUT_OF_MEMORY'
    if t.startswith('PREEMPTED'):
        return 'PREEMPTED'
    if t.startswith('NODE_FAIL'):
        return 'NODE_FAIL'
    return t


def parse_minutes(raw: str) -> Optional[float]:
    s = (raw or '').strip()
    if not s or s in ('N/A', 'Unknown'):
        return None
    days = 0
    if '-' in s:
        day_s, s = s.split('-', 1)
        try:
            days = int(day_s)
        except ValueError:
            return None
    parts = s.split(':')
    try:
        if len(parts) == 3:
            h, m, sec = map(int, parts)
        elif len(parts) == 2:
            h = 0
            m, sec = map(int, parts)
        else:
            return None
    except ValueError:
        return None
    return days * 24 * 60 + h * 60 + m + sec / 60.0


def chunks(seq: List[str], n: int = 120) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def pick(row: Dict[str, str], *keys: str) -> str:
    for k in keys:
        if k in row:
            return (row.get(k) or '').strip()
    return ''


def load_rows() -> List[Row]:
    out: List[Row] = []
    for track, files in TRACK_FILES.items():
        for fp in files:
            with fp.open(newline='') as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    jid = pick(r, 'Job ID', 'job_id', 'JobID')
                    if not jid.isdigit():
                        continue
                    out.append(
                        Row(
                            track=track,
                            csv_source=str(fp.relative_to(REPO)),
                            cluster=pick(r, 'Cluster', 'cluster') or 'Unknown',
                            gpu=pick(r, 'GPU', 'gpu'),
                            node_type=pick(r, 'Node type', 'node_type'),
                            job_name=pick(r, 'Job name', 'job_name'),
                            job_id=jid,
                            qos=pick(r, 'QoS', 'qos'),
                            nodes=pick(r, 'Nodes', 'nodes'),
                            gpus_per_node=pick(r, 'GPUs/node', 'gpus_per_node'),
                            cpus_per_node=pick(r, 'CPUs/node', 'cpus_per_node'),
                            ram_per_node_gib=pick(r, 'RAM/node (GiB)', 'ram_per_node_gib'),
                            status_csv=norm_state(pick(r, 'Status', 'status')),
                            elapsed_csv=pick(r, 'Elapsed job time', 'elapsed_job_time'),
                            timelimit_csv=pick(r, 'Requested time limit', 'requested_time_limit'),
                            data_access_mode=pick(r, 'Data access mode', 'data_access_mode'),
                        )
                    )
    return out


def live_status_maps(rows: List[Row]) -> Dict[Tuple[str, str], Dict[str, str]]:
    ids_by_cluster: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        ids_by_cluster[row.cluster].append(row.job_id)

    out: Dict[Tuple[str, str], Dict[str, str]] = {}

    for cluster, ids in ids_by_cluster.items():
        if not ids:
            continue
        ids = sorted(set(ids))
        runner = run_fal if cluster.lower() == 'falcon' else run_tc

        queue_map: Dict[str, Dict[str, str]] = {}
        for ch in chunks(ids):
            cmd = f"squeue -h -j {','.join(ch)} -o '%A|%T|%M|%l|%S'"
            p = runner(cmd)
            for ln in (p.stdout or '').splitlines():
                ps = ln.split('|')
                if len(ps) < 4:
                    continue
                jid = ps[0].strip()
                if not jid.isdigit():
                    continue
                queue_map[jid] = {
                    'status_live': norm_state(ps[1]),
                    'elapsed_live': ps[2].strip(),
                    'timelimit_live': ps[3].strip(),
                    'start_live': ps[4].strip() if len(ps) > 4 else '',
                    'state_source': 'squeue',
                }

        missing = [jid for jid in ids if jid not in queue_map]
        acct_map: Dict[str, Dict[str, str]] = {}
        for ch in chunks(missing):
            cmd = f"sacct -X -P -n -j {','.join(ch)} --format=JobIDRaw,State,Elapsed,Timelimit,Start,End"
            p = runner(cmd)
            for ln in (p.stdout or '').splitlines():
                ps = ln.split('|')
                if len(ps) < 6:
                    continue
                jid = ps[0].strip()
                if not jid.isdigit() or jid in acct_map:
                    continue
                acct_map[jid] = {
                    'status_live': norm_state(ps[1]),
                    'elapsed_live': ps[2].strip(),
                    'timelimit_live': ps[3].strip(),
                    'start_live': ps[4].strip(),
                    'end_live': ps[5].strip(),
                    'state_source': 'sacct',
                }

        for jid in ids:
            rec = queue_map.get(jid) or acct_map.get(jid) or {
                'status_live': 'UNKNOWN',
                'elapsed_live': '',
                'timelimit_live': '',
                'start_live': '',
                'end_live': '',
                'state_source': 'csv',
            }
            out[(cluster, jid)] = rec

    return out


def parse_metric_value(metric_obj: object, split: str = 'test') -> Optional[float]:
    if metric_obj is None:
        return None
    if isinstance(metric_obj, (int, float)):
        return float(metric_obj)
    if isinstance(metric_obj, dict):
        candidates = [split, f'{split}_mean', f'{split}_avg']
        for key in candidates:
            if key in metric_obj and isinstance(metric_obj[key], (int, float)):
                return float(metric_obj[key])
        return None
    return None


def collect_registry(rows: List[Row], live_map: Dict[Tuple[str, str], Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        live = live_map.get((row.cluster, row.job_id), {})
        status_live = (live.get('status_live') or row.status_csv or 'UNKNOWN')
        elapsed_live = live.get('elapsed_live') or row.elapsed_csv
        timelimit_live = live.get('timelimit_live') or row.timelimit_csv
        elapsed_minutes = parse_minutes(elapsed_live)

        run_dir = RUNS_ROOT / row.job_id
        run_exists = run_dir.is_dir()

        strict_missing = []
        if run_exists:
            for rf in REQUIRED_STRICT:
                if not (run_dir / rf).exists():
                    strict_missing.append(rf)
        else:
            strict_missing = REQUIRED_STRICT.copy()

        metrics_path = run_dir / 'metrics_summary.json'
        metrics_exists = metrics_path.exists()

        mse_test = mae_test = r2_test = None
        data_prefix = ''
        target_names_n = None
        if metrics_exists:
            try:
                with metrics_path.open() as fh:
                    m = json.load(fh)
                mse_test = parse_metric_value(m.get('mse'), 'test')
                mae_test = parse_metric_value(m.get('mae'), 'test')
                r2_test = parse_metric_value(m.get('r2'), 'test')
                meta = m.get('metadata') if isinstance(m.get('metadata'), dict) else {}
                if isinstance(meta, dict):
                    data_prefix = str(meta.get('data_prefix', '') or '')
                    tnames = meta.get('target_names')
                    if isinstance(tnames, list):
                        target_names_n = len(tnames)
            except Exception:
                pass

        artifact_core_valid = bool(metrics_exists)
        artifact_strict_valid = run_exists and not strict_missing
        seeded = '_s' in row.job_name
        if row.job_name.startswith('trn_'):
            run_type = 'training'
        elif row.job_name.startswith('tst_'):
            run_type = 'testing'
        else:
            run_type = 'unknown'

        out.append(
            {
                'track': row.track,
                'csv_source': row.csv_source,
                'cluster': row.cluster,
                'gpu': row.gpu,
                'node_type': row.node_type,
                'job_name': row.job_name,
                'job_id': row.job_id,
                'run_type': run_type,
                'seeded': seeded,
                'qos': row.qos,
                'nodes': row.nodes,
                'gpus_per_node': row.gpus_per_node,
                'cpus_per_node': row.cpus_per_node,
                'ram_per_node_gib': row.ram_per_node_gib,
                'status_csv': row.status_csv,
                'status_live': status_live,
                'state_source': live.get('state_source', 'csv'),
                'elapsed_live': elapsed_live,
                'elapsed_minutes': elapsed_minutes,
                'timelimit_live': timelimit_live,
                'start_live': live.get('start_live', ''),
                'end_live': live.get('end_live', ''),
                'data_access_mode': row.data_access_mode,
                'run_dir_exists': run_exists,
                'artifact_core_valid': artifact_core_valid,
                'artifact_strict_valid': artifact_strict_valid,
                'artifact_strict_missing': ';'.join(strict_missing),
                'metrics_summary_path': str(metrics_path) if metrics_exists else '',
                'mse_test': mse_test,
                'mae_test': mae_test,
                'r2_test': r2_test,
                'metrics_data_prefix': data_prefix,
                'target_names_n': target_names_n,
            }
        )

    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)


def build_status_summary(registry: List[Dict[str, object]]) -> List[Dict[str, object]]:
    c = Counter()
    for r in registry:
        c[(r['track'], r['cluster'], r['status_live'])] += 1
    out = []
    for (track, cluster, status), n in sorted(c.items()):
        out.append({'track': track, 'cluster': cluster, 'status_live': status, 'count': n})
    return out


def build_track_cluster_summary(registry: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in registry:
        grouped[(r['track'], r['cluster'])].append(r)

    out = []
    for (track, cluster), rows in sorted(grouped.items()):
        states = Counter(r['status_live'] for r in rows)
        total = len(rows)
        completed = states['COMPLETED']
        running = states['RUNNING']
        pending = states['PENDING']
        failed_like = sum(states[s] for s in FAILED_LIKE)
        completion_pct = 100.0 * completed / total if total else 0.0
        core_valid_completed = sum(1 for r in rows if r['status_live'] == 'COMPLETED' and r['artifact_core_valid'])
        strict_valid_completed = sum(1 for r in rows if r['status_live'] == 'COMPLETED' and r['artifact_strict_valid'])

        out.append(
            {
                'track': track,
                'cluster': cluster,
                'total_jobs': total,
                'completed': completed,
                'running': running,
                'pending': pending,
                'failed_like': failed_like,
                'completion_pct': round(completion_pct, 2),
                'completed_core_valid': core_valid_completed,
                'completed_strict_valid': strict_valid_completed,
            }
        )
    return out


def summarize_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    mse = [r['mse_test'] for r in rows if isinstance(r['mse_test'], (int, float))]
    mae = [r['mae_test'] for r in rows if isinstance(r['mae_test'], (int, float))]
    r2 = [r['r2_test'] for r in rows if isinstance(r['r2_test'], (int, float))]
    elapsed = [r['elapsed_minutes'] for r in rows if isinstance(r['elapsed_minutes'], (int, float))]

    def stat(v: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if not v:
            return None, None, None, None
        return (min(v), max(v), statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0)

    mse_min, mse_max, mse_mean, mse_std = stat(mse)
    mae_min, mae_max, mae_mean, mae_std = stat(mae)
    r2_min, r2_max, r2_mean, r2_std = stat(r2)
    el_min, el_max, el_mean, el_std = stat(elapsed)

    return {
        'n_rows': len(rows),
        'n_with_metrics': len(mse),
        'mse_min': mse_min,
        'mse_max': mse_max,
        'mse_mean': mse_mean,
        'mse_std': mse_std,
        'mae_min': mae_min,
        'mae_max': mae_max,
        'mae_mean': mae_mean,
        'mae_std': mae_std,
        'r2_min': r2_min,
        'r2_max': r2_max,
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'elapsed_min': el_min,
        'elapsed_max': el_max,
        'elapsed_mean': el_mean,
        'elapsed_std': el_std,
    }


def build_metric_summaries(
    registry: List[Dict[str, object]]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    # only completed + core-valid + has test metrics
    filtered = [
        r for r in registry
        if r['status_live'] == 'COMPLETED' and r['artifact_core_valid'] and isinstance(r['mse_test'], (int, float))
    ]

    by_track_cluster: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    by_track_cluster_type: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    by_gpu_nodes: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)

    for r in filtered:
        by_track_cluster[(r['track'], r['cluster'])].append(r)
        by_track_cluster_type[(r['track'], r['cluster'], r['run_type'])].append(r)
        by_gpu_nodes[(r['track'], r['cluster'], r['gpu'] or 'NA', r['nodes'] or 'NA')].append(r)

    out_tc = []
    for (track, cluster), rows in sorted(by_track_cluster.items()):
        s = summarize_metrics(rows)
        out_tc.append({'track': track, 'cluster': cluster, **s})

    out_tct = []
    for (track, cluster, run_type), rows in sorted(by_track_cluster_type.items()):
        s = summarize_metrics(rows)
        out_tct.append({'track': track, 'cluster': cluster, 'run_type': run_type, **s})

    out_gn = []
    for (track, cluster, gpu, nodes), rows in sorted(by_gpu_nodes.items()):
        s = summarize_metrics(rows)
        out_gn.append({'track': track, 'cluster': cluster, 'gpu': gpu, 'nodes': nodes, **s})

    return out_tc, out_tct, out_gn


def render_table(rows: List[Dict[str, object]], columns: List[str], max_rows: int = 20) -> str:
    if not rows:
        return '(no rows)\n'
    shown = rows[:max_rows]
    out = []
    out.append('| ' + ' | '.join(columns) + ' |')
    out.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for r in shown:
        vals = []
        for c in columns:
            v = r.get(c, '')
            if isinstance(v, float):
                vals.append(f'{v:.6g}')
            else:
                vals.append(str(v))
        out.append('| ' + ' | '.join(vals) + ' |')
    if len(rows) > max_rows:
        out.append(f'\n(Showing first {max_rows} of {len(rows)} rows.)')
    return '\n'.join(out) + '\n'


def write_markdown_report(
    registry: List[Dict[str, object]],
    status_summary: List[Dict[str, object]],
    track_cluster_summary: List[Dict[str, object]],
    metric_tc: List[Dict[str, object]],
    metric_tct: List[Dict[str, object]],
    metric_gn: List[Dict[str, object]],
) -> None:
    p = OUT_DIR / 'analysis_results_report_20260309.md'
    with p.open('w') as fh:
        fh.write('# Four-Track Analysis Results (Automated) — 2026-03-09\n\n')
        fh.write('## Inputs\n')
        for track, files in TRACK_FILES.items():
            fh.write(f'- `{track}`\n')
            for fp in files:
                fh.write(f'  - `{fp.relative_to(REPO)}`\n')
        fh.write('\n')

        fh.write('## Registry Size\n')
        fh.write(f'- Total tracked job rows with numeric IDs: **{len(registry)}**\n\n')

        fh.write('## Track/Cluster Completion Summary\n')
        fh.write(render_table(
            track_cluster_summary,
            ['track', 'cluster', 'total_jobs', 'completed', 'running', 'pending', 'failed_like', 'completion_pct', 'completed_core_valid', 'completed_strict_valid'],
            max_rows=50,
        ))

        fh.write('## Status Distribution\n')
        fh.write(render_table(status_summary, ['track', 'cluster', 'status_live', 'count'], max_rows=80))

        fh.write('## Metric Summary by Track/Cluster (Completed + Core-Valid)\n')
        fh.write(render_table(
            metric_tc,
            ['track', 'cluster', 'n_rows', 'n_with_metrics', 'mse_mean', 'mae_mean', 'r2_mean', 'elapsed_mean', 'mse_std', 'r2_std'],
            max_rows=50,
        ))

        fh.write('## Metric Summary by Track/Cluster/Run Type (Completed + Core-Valid)\n')
        fh.write(render_table(
            metric_tct,
            ['track', 'cluster', 'run_type', 'n_rows', 'n_with_metrics', 'mse_mean', 'mae_mean', 'r2_mean', 'elapsed_mean', 'mse_std', 'r2_std'],
            max_rows=80,
        ))

        fh.write('## Metric Summary by Track/Cluster/GPU/Nodes (Completed + Core-Valid)\n')
        fh.write(render_table(
            metric_gn,
            ['track', 'cluster', 'gpu', 'nodes', 'n_rows', 'mse_mean', 'mae_mean', 'r2_mean', 'elapsed_mean'],
            max_rows=120,
        ))

        fh.write('## Notes\n')
        fh.write('- `core-valid`: `metrics_summary.json` exists in run directory.\n')
        fh.write('- `strict-valid`: required artifact set exists (`params.yaml`, `split_indices.npz`, `net.txt`, `metrics_summary.json`, `metrics_summary.csv`, `predictions.npz`).\n')


def main() -> None:
    rows = load_rows()
    live = live_status_maps(rows)
    registry = collect_registry(rows, live)

    write_csv(OUT_DIR / 'master_registry_20260309.csv', registry)

    status_summary = build_status_summary(registry)
    write_csv(OUT_DIR / 'status_summary_20260309.csv', status_summary)

    track_cluster_summary = build_track_cluster_summary(registry)
    write_csv(OUT_DIR / 'track_cluster_summary_20260309.csv', track_cluster_summary)

    metric_tc, metric_tct, metric_gn = build_metric_summaries(registry)
    write_csv(OUT_DIR / 'metric_summary_track_cluster_20260309.csv', metric_tc)
    write_csv(OUT_DIR / 'metric_summary_track_cluster_run_type_20260309.csv', metric_tct)
    write_csv(OUT_DIR / 'metric_summary_gpu_nodes_20260309.csv', metric_gn)

    write_markdown_report(registry, status_summary, track_cluster_summary, metric_tc, metric_tct, metric_gn)

    print('WROTE', OUT_DIR)
    print('ROWS', len(registry))


if __name__ == '__main__':
    main()
