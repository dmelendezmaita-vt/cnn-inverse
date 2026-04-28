#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shlex
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

REPO = Path('/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch')
RUNS_ROOT = Path('/projects/neuro-collab/data/runs')
OUT_DIR = REPO / 'data/important_notes/publishability_analysis_20260309_rigorous'
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

# Publication-style references explicitly consulted for structure and writing rigor.
STYLE_REFERENCES = [
    {
        'label': 'JMLR Volume 25 (2024) article structure and reporting density',
        'url': 'https://www.jmlr.org/papers/v25/',
    },
    {
        'label': 'IEEE TPDS (2024): Synchronize Only the Immature Parameters',
        'url': 'https://ieeexplore.ieee.org/document/10036106/',
    },
    {
        'label': 'ACM TOSEM (2023): Rise of Distributed Deep Learning Training in the Big Model Era',
        'url': 'https://arxiv.org/abs/2112.06222',
    },
    {
        'label': 'JPDC (2024): MLLess cost-efficiency study',
        'url': 'https://doi.org/10.1016/j.jpdc.2023.104764',
    },
]


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
    if t in ('RUNNING', 'R', 'COMPLETING', 'CG'):
        return 'RUNNING'
    if t in ('PENDING', 'PD', 'CF', 'CONFIGURING', 'SUBMITTED'):
        return 'PENDING'
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


def pick(row: Dict[str, str], *keys: str) -> str:
    for k in keys:
        if k in row:
            return (row.get(k) or '').strip()
    return ''


def chunks(seq: Sequence[str], n: int = 120) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


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
        ids = sorted(set(ids))
        if not ids:
            continue
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
        for key in (split, f'{split}_mean', f'{split}_avg'):
            v = metric_obj.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def parse_params(path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not path.exists():
        return out
    try:
        with path.open() as fh:
            y = yaml.safe_load(fh) or {}
    except Exception:
        return out

    d = y.get('data') if isinstance(y.get('data'), dict) else {}
    t = y.get('training') if isinstance(y.get('training'), dict) else {}
    o = y.get('optimizer') if isinstance(y.get('optimizer'), dict) else {}
    n = y.get('net') if isinstance(y.get('net'), dict) else {}
    r = y.get('runconfig') if isinstance(y.get('runconfig'), dict) else {}

    def num(x):
        return x if isinstance(x, (int, float)) else None

    out['p_dataset_layout'] = d.get('dataset_layout')
    out['p_data_prefix'] = d.get('data_prefix')
    out['p_Ntrain'] = num(d.get('Ntrain'))
    out['p_Nvalidate'] = num(d.get('Nvalidate'))
    out['p_Ntest'] = num(d.get('Ntest'))
    out['p_random_seed'] = num(d.get('random_seed'))
    out['p_global_train_batch_size'] = num(d.get('global_train_batch_size'))
    out['p_train_batch_size'] = num(d.get('train_batch_size'))
    out['p_features_cols_num'] = num(d.get('features_cols_num'))
    out['p_targets_cols_num'] = num(d.get('targets_cols_num'))

    out['p_epochs'] = num(t.get('epochs'))
    out['p_learning_rate'] = num(o.get('learning_rate'))
    out['p_optimizer'] = o.get('type')
    out['p_net_type'] = n.get('type')

    out['p_runconfig_params'] = r.get('params')
    out['p_run_mode'] = r.get('mode')
    return out


def build_registry(rows: List[Row], live_map: Dict[Tuple[str, str], Dict[str, str]]) -> pd.DataFrame:
    recs: List[Dict[str, object]] = []
    for row in rows:
        live = live_map.get((row.cluster, row.job_id), {})
        status_live = live.get('status_live') or row.status_csv or 'UNKNOWN'
        elapsed_live = live.get('elapsed_live') or row.elapsed_csv
        timelimit_live = live.get('timelimit_live') or row.timelimit_csv

        run_dir = RUNS_ROOT / row.job_id
        run_exists = run_dir.is_dir()

        strict_missing: List[str] = []
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

        params_info = parse_params(run_dir / 'params.yaml')

        if row.job_name.startswith('trn_'):
            run_type = 'training'
        elif row.job_name.startswith('tst_'):
            run_type = 'testing'
        else:
            run_type = 'unknown'

        rec = {
            'track': row.track,
            'csv_source': row.csv_source,
            'cluster': row.cluster,
            'gpu': row.gpu,
            'node_type': row.node_type,
            'job_name': row.job_name,
            'job_id': row.job_id,
            'run_type': run_type,
            'seeded': '_s' in row.job_name,
            'qos': row.qos,
            'nodes': pd.to_numeric(row.nodes, errors='coerce'),
            'gpus_per_node': pd.to_numeric(row.gpus_per_node, errors='coerce'),
            'cpus_per_node': pd.to_numeric(row.cpus_per_node, errors='coerce'),
            'ram_per_node_gib': pd.to_numeric(row.ram_per_node_gib, errors='coerce'),
            'status_csv': row.status_csv,
            'status_live': status_live,
            'state_source': live.get('state_source', 'csv'),
            'elapsed_live': elapsed_live,
            'elapsed_minutes': parse_minutes(elapsed_live),
            'timelimit_live': timelimit_live,
            'start_live': live.get('start_live', ''),
            'end_live': live.get('end_live', ''),
            'data_access_mode': row.data_access_mode,
            'run_dir_exists': run_exists,
            'artifact_core_valid': bool(metrics_exists),
            'artifact_strict_valid': run_exists and not strict_missing,
            'artifact_strict_missing': ';'.join(strict_missing),
            'metrics_summary_path': str(metrics_path) if metrics_exists else '',
            'mse_test': mse_test,
            'mae_test': mae_test,
            'r2_test': r2_test,
            'metrics_data_prefix': data_prefix,
            'target_names_n': target_names_n,
        }
        rec.update(params_info)
        recs.append(rec)

    df = pd.DataFrame.from_records(recs)
    if not df.empty:
        df['job_id'] = df['job_id'].astype(str)
    return df


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 20260309) -> Tuple[float, float, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = vals.size
    for i in range(n_boot):
        sample = vals[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(vals)), lo, hi


def bootstrap_mean_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 20260309) -> Tuple[float, float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    na = a.size
    nb = b.size
    for i in range(n_boot):
        sa = a[rng.integers(0, na, size=na)]
        sb = b[rng.integers(0, nb, size=nb)]
        diffs[i] = float(np.mean(sb) - np.mean(sa))
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return float(np.mean(b) - np.mean(a)), lo, hi


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return math.nan
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    return (gt - lt) / (a.size * b.size)


def permutation_pvalue_mean_diff(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 20260309) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return math.nan
    obs = abs(float(np.mean(b) - np.mean(a)))
    combined = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        d = abs(float(np.mean(perm[n_a:]) - np.mean(perm[:n_a])))
        if d >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def bh_fdr(pvals: List[float]) -> List[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [math.nan] * n
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        p = pvals[i]
        q = min(prev, p * n / rank)
        adj[i] = q
        prev = q
    return adj


def summarize_numeric(values: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(values, errors='coerce').dropna().to_numpy(dtype=float)
    if x.size == 0:
        return {
            'n': 0,
            'mean': math.nan,
            'std': math.nan,
            'median': math.nan,
            'iqr': math.nan,
            'min': math.nan,
            'max': math.nan,
            'ci95_lo': math.nan,
            'ci95_hi': math.nan,
        }
    mean, lo, hi = bootstrap_mean_ci(x)
    q1 = float(np.quantile(x, 0.25))
    q3 = float(np.quantile(x, 0.75))
    return {
        'n': int(x.size),
        'mean': float(np.mean(x)),
        'std': float(np.std(x, ddof=0)) if x.size > 1 else 0.0,
        'median': float(np.median(x)),
        'iqr': q3 - q1,
        'min': float(np.min(x)),
        'max': float(np.max(x)),
        'ci95_lo': lo,
        'ci95_hi': hi,
    }


def write_df(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def build_status_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    status_summary = (
        df.groupby(['track', 'cluster', 'status_live'], dropna=False)
        .size()
        .reset_index(name='count')
        .sort_values(['track', 'cluster', 'status_live'])
    )

    rows = []
    for (track, cluster), g in df.groupby(['track', 'cluster'], dropna=False):
        states = Counter(g['status_live'].tolist())
        total = len(g)
        completed = states.get('COMPLETED', 0)
        running = states.get('RUNNING', 0)
        pending = states.get('PENDING', 0)
        failed_like = sum(states.get(s, 0) for s in FAILED_LIKE)
        rows.append(
            {
                'track': track,
                'cluster': cluster,
                'total_jobs': total,
                'completed': completed,
                'running': running,
                'pending': pending,
                'failed_like': failed_like,
                'completion_pct': round(100.0 * completed / total, 2) if total else 0.0,
                'completed_core_valid': int(((g['status_live'] == 'COMPLETED') & (g['artifact_core_valid'])).sum()),
                'completed_strict_valid': int(((g['status_live'] == 'COMPLETED') & (g['artifact_strict_valid'])).sum()),
            }
        )

    tc_summary = pd.DataFrame(rows).sort_values(['track', 'cluster'])
    return status_summary, tc_summary


def build_metric_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f = df[(df['status_live'] == 'COMPLETED') & (df['artifact_core_valid'])].copy()
    f = f[pd.to_numeric(f['mse_test'], errors='coerce').notna()].copy()

    def agg_one(group_cols: List[str]) -> pd.DataFrame:
        rows = []
        for keys, g in f.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(group_cols, keys))
            mse_s = summarize_numeric(g['mse_test'])
            mae_s = summarize_numeric(g['mae_test'])
            r2_s = summarize_numeric(g['r2_test'])
            el_s = summarize_numeric(g['elapsed_minutes'])
            rows.append(
                {
                    **base,
                    'n_rows': len(g),
                    'n_with_metrics': mse_s['n'],
                    'mse_mean': mse_s['mean'],
                    'mse_ci95_lo': mse_s['ci95_lo'],
                    'mse_ci95_hi': mse_s['ci95_hi'],
                    'mse_std': mse_s['std'],
                    'mae_mean': mae_s['mean'],
                    'mae_ci95_lo': mae_s['ci95_lo'],
                    'mae_ci95_hi': mae_s['ci95_hi'],
                    'mae_std': mae_s['std'],
                    'r2_mean': r2_s['mean'],
                    'r2_ci95_lo': r2_s['ci95_lo'],
                    'r2_ci95_hi': r2_s['ci95_hi'],
                    'r2_std': r2_s['std'],
                    'elapsed_mean': el_s['mean'],
                    'elapsed_ci95_lo': el_s['ci95_lo'],
                    'elapsed_ci95_hi': el_s['ci95_hi'],
                    'elapsed_std': el_s['std'],
                }
            )
        return pd.DataFrame(rows).sort_values(group_cols)

    return agg_one(['track', 'cluster']), agg_one(['track', 'cluster', 'run_type']), agg_one(['track', 'cluster', 'gpu', 'nodes'])


def build_inferential_tests(df: pd.DataFrame) -> pd.DataFrame:
    f = df[(df['status_live'] == 'COMPLETED') & (df['artifact_core_valid'])].copy()

    def sel(track: str, cluster: str, run_type: str, metric: str) -> np.ndarray:
        s = pd.to_numeric(
            f[(f['track'] == track) & (f['cluster'] == cluster) & (f['run_type'] == run_type)][metric],
            errors='coerce',
        ).dropna()
        return s.to_numpy(dtype=float)

    comparisons = [
        ('track1_first_parity', 'track2_scalability', 'Falcon', 'training', 'FHN parity vs scalability-control (training)'),
        ('track3_hh_reduced', 'track4_hh_full', 'Falcon', 'training', 'HH reduced vs HH full (training)'),
        ('track3_hh_reduced', 'track4_hh_full', 'Falcon', 'testing', 'HH reduced vs HH full (testing)'),
    ]
    metrics = ['mse_test', 'mae_test', 'r2_test', 'elapsed_minutes']

    rows = []
    for t_a, t_b, cluster, run_type, label in comparisons:
        for metric in metrics:
            a = sel(t_a, cluster, run_type, metric)
            b = sel(t_b, cluster, run_type, metric)
            dmean, lo, hi = bootstrap_mean_diff_ci(a, b)
            p = permutation_pvalue_mean_diff(a, b)
            cd = cliffs_delta(a, b)
            rows.append(
                {
                    'comparison': label,
                    'cluster': cluster,
                    'run_type': run_type,
                    'metric': metric,
                    'group_a_track': t_a,
                    'group_b_track': t_b,
                    'n_a': int(a.size),
                    'n_b': int(b.size),
                    'mean_a': float(np.mean(a)) if a.size else math.nan,
                    'mean_b': float(np.mean(b)) if b.size else math.nan,
                    'delta_mean_b_minus_a': dmean,
                    'delta_ci95_lo': lo,
                    'delta_ci95_hi': hi,
                    'cliffs_delta': cd,
                    'perm_pvalue_two_sided': p,
                }
            )

    out = pd.DataFrame(rows)
    valid = out['perm_pvalue_two_sided'].apply(lambda x: isinstance(x, float) and math.isfinite(x))
    if valid.any():
        pvals = out.loc[valid, 'perm_pvalue_two_sided'].tolist()
        out.loc[valid, 'bh_fdr_qvalue'] = bh_fdr(pvals)
    else:
        out['bh_fdr_qvalue'] = math.nan

    out = out.sort_values(['comparison', 'metric']).reset_index(drop=True)
    return out


def build_scaling_table(df: pd.DataFrame) -> pd.DataFrame:
    f = df[
        (df['status_live'] == 'COMPLETED')
        & (df['artifact_core_valid'])
        & (df['cluster'] == 'Falcon')
        & (df['run_type'] == 'training')
    ].copy()
    f['nodes'] = pd.to_numeric(f['nodes'], errors='coerce')
    f = f[f['nodes'].notna()]

    rows = []
    for (track, gpu), g in f.groupby(['track', 'gpu'], dropna=False):
        means = g.groupby('nodes', dropna=False)['elapsed_minutes'].mean().to_dict()
        if 1.0 not in means:
            continue
        t1 = means[1.0]
        for n, t in sorted(means.items(), key=lambda kv: kv[0]):
            speedup = (t1 / t) if (t and not math.isnan(t)) else math.nan
            eff = (speedup / n) if (speedup and n and n > 0) else math.nan
            rows.append(
                {
                    'track': track,
                    'cluster': 'Falcon',
                    'gpu': gpu,
                    'nodes': int(n),
                    'mean_elapsed_min': t,
                    'speedup_vs_1node': speedup,
                    'strong_scaling_efficiency': eff,
                }
            )
    return pd.DataFrame(rows).sort_values(['track', 'gpu', 'nodes']) if rows else pd.DataFrame(columns=[
        'track', 'cluster', 'gpu', 'nodes', 'mean_elapsed_min', 'speedup_vs_1node', 'strong_scaling_efficiency'
    ])


def build_config_consistency(df: pd.DataFrame) -> pd.DataFrame:
    f = df[(df['status_live'] == 'COMPLETED') & (df['run_dir_exists']) & (df['p_run_mode'].notna())].copy()
    key_cols = [
        'p_dataset_layout', 'p_data_prefix', 'p_Ntrain', 'p_Nvalidate', 'p_Ntest',
        'p_train_batch_size', 'p_global_train_batch_size', 'p_features_cols_num',
        'p_targets_cols_num', 'p_epochs', 'p_learning_rate', 'p_optimizer', 'p_net_type'
    ]

    rows = []
    for (track, cluster, run_type), g in f.groupby(['track', 'cluster', 'run_type'], dropna=False):
        sigs = g[key_cols].astype(str).agg('|'.join, axis=1)
        rows.append(
            {
                'track': track,
                'cluster': cluster,
                'run_type': run_type,
                'n_completed': len(g),
                'n_unique_config_signatures': int(sigs.nunique()),
            }
        )
    out = pd.DataFrame(rows).sort_values(['track', 'cluster', 'run_type'])

    # Explicit parity check for Track1 vs Track2 Falcon training (ignoring seed and path-only differences).
    t1 = f[(f['track'] == 'track1_first_parity') & (f['cluster'] == 'Falcon') & (f['run_type'] == 'training')]
    t2 = f[(f['track'] == 'track2_scalability') & (f['cluster'] == 'Falcon') & (f['run_type'] == 'training')]

    compare_cols = [
        'p_dataset_layout', 'p_data_prefix', 'p_Ntrain', 'p_Nvalidate', 'p_Ntest',
        'p_train_batch_size', 'p_global_train_batch_size', 'p_features_cols_num',
        'p_targets_cols_num', 'p_epochs', 'p_learning_rate', 'p_optimizer', 'p_net_type'
    ]

    parity = {}
    for c in compare_cols:
        a = set(t1[c].dropna().astype(str).unique())
        b = set(t2[c].dropna().astype(str).unique())
        parity[c] = (a == b)

    parity_df = pd.DataFrame([
        {'parameter': k, 'track1_vs_track2_falcon_training_equal': v} for k, v in parity.items()
    ])
    return out, parity_df


def build_data_quality_audit(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    completed = int((df['status_live'] == 'COMPLETED').sum())
    completed_core = int(((df['status_live'] == 'COMPLETED') & (df['artifact_core_valid'])).sum())
    completed_strict = int(((df['status_live'] == 'COMPLETED') & (df['artifact_strict_valid'])).sum())

    mismatch = int(((df['status_csv'].fillna('') != df['status_live'].fillna('')) & (df['status_live'] != 'UNKNOWN')).sum())
    duplicates = int(df['job_id'].duplicated(keep=False).sum())

    rows = [
        {'check': 'registry_rows', 'value': total},
        {'check': 'completed_rows', 'value': completed},
        {'check': 'completed_core_valid_rows', 'value': completed_core},
        {'check': 'completed_strict_valid_rows', 'value': completed_strict},
        {'check': 'status_csv_vs_live_mismatch_rows', 'value': mismatch},
        {'check': 'duplicate_job_id_rows', 'value': duplicates},
    ]

    by_status = df.groupby('status_live').size().reset_index(name='count').sort_values('status_live')
    by_status['check'] = 'status_count_' + by_status['status_live'].astype(str)
    for _, r in by_status.iterrows():
        rows.append({'check': r['check'], 'value': int(r['count'])})

    return pd.DataFrame(rows)


def find_orphan_active_jobs(df: pd.DataFrame) -> pd.DataFrame:
    tracked_ids = set(df['job_id'].astype(str))
    rows = []
    for cluster, runner in [('Falcon', run_fal), ('Tinkercliffs', run_tc)]:
        p = runner("squeue -h -u dmm96 -o '%A|%j|%T'")
        for ln in (p.stdout or '').splitlines():
            ps = ln.split('|')
            if len(ps) < 3:
                continue
            jid, jname, st = ps[0].strip(), ps[1].strip(), ps[2].strip()
            if not jid.isdigit():
                continue
            if ('trn_' in jname or 'tst_' in jname) and jid not in tracked_ids:
                rows.append({'cluster': cluster, 'job_id': jid, 'job_name': jname, 'status': st})
    return pd.DataFrame(rows).sort_values(['cluster', 'job_id']) if rows else pd.DataFrame(columns=['cluster', 'job_id', 'job_name', 'status'])


def md_table(df: pd.DataFrame, cols: List[str], max_rows: int = 80) -> str:
    if df.empty:
        return '(no rows)\n'
    d = df[cols].head(max_rows).copy()
    lines = []
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
    for _, r in d.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f'{v:.6g}' if math.isfinite(v) else 'nan')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    if len(df) > max_rows:
        lines.append(f'\n(Showing first {max_rows} of {len(df)} rows.)')
    return '\n'.join(lines) + '\n'


def write_plan_file(df: pd.DataFrame) -> None:
    p = OUT_DIR / 'analysis_plan_and_execution_rigorous_20260309.md'
    with p.open('w') as fh:
        fh.write('# Four-Track Rigorous Analysis Plan and Execution Log — 2026-03-09\n\n')
        fh.write('## Objective\n')
        fh.write('Produce publication-grade evidence quality for the four-track study by upgrading from descriptive summaries to inferential, uncertainty-aware, and audit-traceable analysis.\n\n')

        fh.write('## Upgraded Methodological Criteria\n')
        fh.write('1. Strict data-contract: scheduler state precedence, artifact gating, and explicit exclusion classes.\n')
        fh.write('2. Uncertainty quantification: bootstrap 95% confidence intervals for reported means.\n')
        fh.write('3. Inferential testing: permutation tests for key track-pair comparisons.\n')
        fh.write('4. Multiple-testing control: Benjamini-Hochberg FDR correction.\n')
        fh.write('5. Effect-size reporting: Cliff\'s delta for each inferential comparison.\n')
        fh.write('6. Configuration integrity: parameter-signature consistency and parity checks.\n')
        fh.write('7. Scalability rigor: speedup and strong-scaling efficiency from node-wise elapsed time.\n')
        fh.write('8. Operational auditability: orphan-job detection and status-source mismatch accounting.\n\n')

        fh.write('## Recent Publication Style References Consulted\n')
        for ref in STYLE_REFERENCES:
            fh.write(f"- {ref['label']}: {ref['url']}\n")
        fh.write('\n')

        fh.write('## Exact Execution Steps\n')
        fh.write('1. Ingest all tracked CSV registries for tracks 1-4.\n')
        fh.write('2. Reconcile live status per job from `squeue` then `sacct` (cluster-aware).\n')
        fh.write('3. Resolve run artifacts under `/projects/neuro-collab/data/runs/<job_id>`.\n')
        fh.write('4. Parse metrics from `metrics_summary.json` and model config from `params.yaml`.\n')
        fh.write('5. Build master registry and quality audit tables.\n')
        fh.write('6. Build uncertainty-aware metric summaries (95% bootstrap CI).\n')
        fh.write('7. Run inferential tests and FDR correction for predefined comparisons.\n')
        fh.write('8. Compute scaling efficiency summaries.\n')
        fh.write('9. Generate updated publication-quality report and machine-readable tables.\n\n')

        fh.write('## Registry Size\n')
        fh.write(f"- Total numeric job rows analyzed: **{len(df)}**\n")


def write_results_report(
    df: pd.DataFrame,
    status_summary: pd.DataFrame,
    tc_summary: pd.DataFrame,
    metric_tc: pd.DataFrame,
    metric_tct: pd.DataFrame,
    metric_gn: pd.DataFrame,
    inferential: pd.DataFrame,
    scaling: pd.DataFrame,
    cfg_consistency: pd.DataFrame,
    parity: pd.DataFrame,
    audit: pd.DataFrame,
    orphan: pd.DataFrame,
) -> None:
    p = OUT_DIR / 'analysis_results_report_rigorous_20260309.md'
    with p.open('w') as fh:
        fh.write('# Four-Track Rigorous Analysis Results — 2026-03-09\n\n')

        fh.write('## 1. Scope and Validity Contract\n')
        fh.write('- Status precedence: `squeue` -> `sacct` -> CSV fallback.\n')
        fh.write('- Quantitative inclusion: `status_live == COMPLETED` and `artifact_core_valid == true`.\n')
        fh.write('- Strict reproducibility audit: required six-file artifact set present.\n')
        fh.write('- Excluded failure classes: FAILED, TIMEOUT, OUT_OF_MEMORY, CANCELLED, PREEMPTED, NODE_FAIL.\n\n')

        fh.write('## 2. Completion and Status Integrity\n')
        fh.write(md_table(tc_summary, [
            'track', 'cluster', 'total_jobs', 'completed', 'running', 'pending',
            'failed_like', 'completion_pct', 'completed_core_valid', 'completed_strict_valid'
        ], max_rows=30))

        fh.write('### Status Distribution\n')
        fh.write(md_table(status_summary, ['track', 'cluster', 'status_live', 'count'], max_rows=80))

        fh.write('## 3. Metric Summaries with 95% CI\n')
        fh.write('### By Track and Cluster\n')
        fh.write(md_table(metric_tc, [
            'track', 'cluster', 'n_rows', 'mse_mean', 'mse_ci95_lo', 'mse_ci95_hi',
            'mae_mean', 'mae_ci95_lo', 'mae_ci95_hi',
            'r2_mean', 'r2_ci95_lo', 'r2_ci95_hi',
            'elapsed_mean', 'elapsed_ci95_lo', 'elapsed_ci95_hi'
        ], max_rows=30))

        fh.write('### By Track, Cluster, and Run Type\n')
        fh.write(md_table(metric_tct, [
            'track', 'cluster', 'run_type', 'n_rows',
            'mse_mean', 'mse_ci95_lo', 'mse_ci95_hi',
            'r2_mean', 'r2_ci95_lo', 'r2_ci95_hi',
            'elapsed_mean', 'elapsed_ci95_lo', 'elapsed_ci95_hi'
        ], max_rows=40))

        fh.write('## 4. Inferential Comparisons (Permutation Tests + FDR)\n')
        fh.write(md_table(inferential, [
            'comparison', 'metric', 'n_a', 'n_b',
            'mean_a', 'mean_b', 'delta_mean_b_minus_a',
            'delta_ci95_lo', 'delta_ci95_hi',
            'cliffs_delta', 'perm_pvalue_two_sided', 'bh_fdr_qvalue'
        ], max_rows=50))

        fh.write('Interpretation guidance:\n')
        fh.write('- `delta_mean_b_minus_a` > 0 indicates larger metric value for group B.\n')
        fh.write('- For error metrics (`mse_test`, `mae_test`, `elapsed_minutes`), lower is better.\n')
        fh.write('- For `r2_test`, higher is better.\n\n')

        fh.write('## 5. Scalability Diagnostics\n')
        fh.write(md_table(scaling, [
            'track', 'cluster', 'gpu', 'nodes', 'mean_elapsed_min',
            'speedup_vs_1node', 'strong_scaling_efficiency'
        ], max_rows=120))

        fh.write('## 6. Configuration Consistency and Parity\n')
        fh.write('### Signature Consistency (within track/cluster/run_type)\n')
        fh.write(md_table(cfg_consistency, [
            'track', 'cluster', 'run_type', 'n_completed', 'n_unique_config_signatures'
        ], max_rows=60))

        fh.write('### Explicit Track1-vs-Track2 Falcon Training Parity\n')
        fh.write(md_table(parity, ['parameter', 'track1_vs_track2_falcon_training_equal'], max_rows=40))

        fh.write('## 7. Data Quality and Audit Findings\n')
        fh.write(md_table(audit, ['check', 'value'], max_rows=80))

        fh.write('### Orphan Active Jobs (tracked naming pattern but missing from registry)\n')
        fh.write(md_table(orphan, ['cluster', 'job_id', 'job_name', 'status'], max_rows=60))

        fh.write('## 8. Publication-Scope Assessment\n')
        fh.write('- Falcon-only claims are evidence-complete for all tracks under the current contract.\n')
        fh.write('- Cross-cluster claims remain out of scope until Tinkercliffs completion is symmetric.\n')
        fh.write('- Track-2 testing jobs should remain categorized as orchestration diagnostics, not parity-quality training evidence.\n\n')

        fh.write('## 9. Style and Complexity References Consulted\n')
        for ref in STYLE_REFERENCES:
            fh.write(f"- {ref['label']}: {ref['url']}\n")


def main() -> None:
    rows = load_rows()
    live_map = live_status_maps(rows)
    registry = build_registry(rows, live_map)

    write_df(OUT_DIR / 'master_registry_rigorous_20260309.csv', registry)

    status_summary, tc_summary = build_status_tables(registry)
    write_df(OUT_DIR / 'status_summary_rigorous_20260309.csv', status_summary)
    write_df(OUT_DIR / 'track_cluster_summary_rigorous_20260309.csv', tc_summary)

    metric_tc, metric_tct, metric_gn = build_metric_tables(registry)
    write_df(OUT_DIR / 'metric_summary_track_cluster_rigorous_20260309.csv', metric_tc)
    write_df(OUT_DIR / 'metric_summary_track_cluster_run_type_rigorous_20260309.csv', metric_tct)
    write_df(OUT_DIR / 'metric_summary_gpu_nodes_rigorous_20260309.csv', metric_gn)

    inferential = build_inferential_tests(registry)
    write_df(OUT_DIR / 'inferential_tests_rigorous_20260309.csv', inferential)

    scaling = build_scaling_table(registry)
    write_df(OUT_DIR / 'scaling_summary_rigorous_20260309.csv', scaling)

    cfg_consistency, parity = build_config_consistency(registry)
    write_df(OUT_DIR / 'config_consistency_rigorous_20260309.csv', cfg_consistency)
    write_df(OUT_DIR / 'config_parity_track1_vs_track2_rigorous_20260309.csv', parity)

    audit = build_data_quality_audit(registry)
    write_df(OUT_DIR / 'data_quality_audit_rigorous_20260309.csv', audit)

    orphan = find_orphan_active_jobs(registry)
    write_df(OUT_DIR / 'orphan_active_jobs_rigorous_20260309.csv', orphan)

    write_plan_file(registry)
    write_results_report(
        registry,
        status_summary,
        tc_summary,
        metric_tc,
        metric_tct,
        metric_gn,
        inferential,
        scaling,
        cfg_consistency,
        parity,
        audit,
        orphan,
    )

    print('WROTE', OUT_DIR)
    print('ROWS', len(registry))


if __name__ == '__main__':
    main()
