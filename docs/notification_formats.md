# Notification Format Options

This document stores the finalized notification format options for SLURM job reporting.

## Formatting Rules (Applied to All Options)

- Percentages are integers only (no decimals).
- RAM is reported as `used/allocated GiB` with integer values.
- VRAM is reported as `used/total GiB` with integer values (not percentage-only).
- `GPU util per GPU` always includes `%` on every value.

## Information Required In All Options

All options include:

- Process name
- Job ID
- Mode
- Run start timestamp (local cluster time)
- Node count
- GPUs per node
- CPUs per node
- Status
- Per-node CPU utilization (`%`)
- Per-node RAM utilization (`used/allocated GiB` + `%`)
- Per-node GPU utilization (per-GPU `%` list)
- Per-node VRAM usage (`GiB/GiB`)
- Error section (`none` or explicit error details)

## Per-Option Differences (What Is Omitted)

Reference: "fullest schema" means everything above plus cluster-level aggregate lines and structured error keys (`code`, `node`, `gpu`, `step`, `message`).

| Option | Omitted vs fullest schema |
|---|---|
| 1 | Cluster-level aggregate block; structured error keys |
| 2 | Cluster-level aggregate block; structured error keys |
| 3 | Cluster-level aggregate block; structured error keys |
| 4 | Cluster-level aggregate block; structured error keys |
| 5 | Cluster-level aggregate block; structured error keys |
| 6 | Per-GPU VRAM split is summarized at node/cluster level (not listed GPU-by-GPU); structured error keys unless failure section is expanded |
| 7 | Cluster-level aggregate block; structured error keys |
| 8 | Cluster-level aggregate block (focuses on node-level + rich error when present) |

## Option 1: Structured Card + Node Blocks

```text
[RUN STATUS]
Process        : fhn_train
Job ID         : 4837219
Mode           : train
Start (local)  : 2026-03-01 09:12:44 EST
Nodes          : 2 (node01,node02)
GPUs / node    : 4
CPUs / node    : 64
Status         : RUNNING

[PER-NODE UTILIZATION]
node01 | CPU 71% | RAM 201/240 GiB (84%) | GPU util [82%,76%,79%,81%] | VRAM [51/80,49/80,50/80,52/80 GiB]
node02 | CPU 58% | RAM 188/240 GiB (78%) | GPU util [68%,71%,66%,70%] | VRAM [42/80,45/80,42/80,46/80 GiB]

[ERROR]
- none
```

## Option 2: Structured Card + Table

```text
[RUN STATUS]
Process: fhn_train | Job ID: 4837219 | Mode: train | Start(local): 2026-03-01 09:12:44 EST | Status: RUNNING
Nodes: 2 | GPUs/node: 4 | CPUs/node: 64

[NODE RESOURCE TABLE]
Node    CPU    RAM (GiB)            GPU util per GPU         VRAM per GPU (GiB)
node01  71%    201/240 (84%)        [82%,76%,79%,81%]        [51/80,49/80,50/80,52/80]
node02  58%    188/240 (78%)        [68%,71%,66%,70%]        [42/80,45/80,42/80,46/80]

[ERROR]
- none
```

## Option 3: Sectioned Ops Report

```text
RUN
- process: fhn_train
- job_id: 4837219
- mode: train
- start_local: 2026-03-01 09:12:44 EST
- status: RUNNING

ALLOC
- nodes: 2
- gpus_per_node: 4
- cpus_per_node: 64

UTILIZATION (CURRENT)
- node01: cpu=71%, ram=201/240 GiB (84%), gpu=[82%,76%,79%,81%], vram=[51/80,49/80,50/80,52/80 GiB]
- node02: cpu=58%, ram=188/240 GiB (78%), gpu=[68%,71%,66%,70%], vram=[42/80,45/80,42/80,46/80 GiB]

ERROR
- none
```

## Option 4: Summary + Detailed Node Lines

```text
[SUMMARY]
fhn_train | job=4837219 | mode=train | start=2026-03-01 09:12:44 EST | nodes=2 | gpn=4 | cpn=64 | status=RUNNING

[NODE DETAILS]
node01: CPU 71% | RAM 201/240 GiB (84%) | GPU util g0=82% g1=76% g2=79% g3=81% | VRAM g0=51/80 g1=49/80 g2=50/80 g3=52/80 GiB
node02: CPU 58% | RAM 188/240 GiB (78%) | GPU util g0=68% g1=71% g2=66% g3=70% | VRAM g0=42/80 g1=45/80 g2=42/80 g3=46/80 GiB

[ERROR]
- none
```

## Option 5: Wide Matrix

```text
[RUN STATUS]
Process=fhn_train  JobID=4837219  Mode=train  Start=2026-03-01 09:12:44 EST  Status=RUNNING
Nodes=2  GPUs/node=4  CPUs/node=64

Node    CPU    RAM (GiB)             GPU util (g0,g1,g2,g3)      VRAM (GiB, g0,g1,g2,g3)
node01  71%    201/240 (84%)         82%,76%,79%,81%             51/80,49/80,50/80,52/80
node02  58%    188/240 (78%)         68%,71%,66%,70%             42/80,45/80,42/80,46/80

[ERROR]
- none
```

## Option 6: Two-Tier Card (Current Selection)

```text
[RUN STATUS]
Process        : fhn_train
Job ID         : 4837219
Mode           : train
Start (local)  : 2026-03-01 09:12:44 EST
Nodes          : 2
GPUs / node    : 4
CPUs / node    : 64
Status         : RUNNING

[CLUSTER CURRENT]
CPU avg        : 65%
RAM            : 389/480 GiB (81%)
GPU util avg   : 74%
VRAM avg       : 377/640 GiB (59%)

[PER NODE]
node01         : CPU 71% | RAM 201/240 GiB (84%) | GPU util avg 80% [82%,76%,79%,81%] | VRAM 202/320 GiB (63%)
node02         : CPU 58% | RAM 188/240 GiB (78%) | GPU util avg 69% [68%,71%,66%,70%] | VRAM 175/320 GiB (55%)

[ERROR]
- none
```

## Option 7: Key-Value + Node Maps

```text
[RUN STATUS]
process=fhn_train
job_id=4837219
mode=train
start_local="2026-03-01 09:12:44 EST"
nodes=2
gpus_per_node=4
cpus_per_node=64
status=RUNNING

[UTILIZATION]
node01 cpu=71% ram=201/240GiB(84%) gpu_util=[82%,76%,79%,81%] vram_gib=[51/80,49/80,50/80,52/80]
node02 cpu=58% ram=188/240GiB(78%) gpu_util=[68%,71%,66%,70%] vram_gib=[42/80,45/80,42/80,46/80]

[ERROR]
- none
```

## Option 8: Structured Card + Detailed Error Record

```text
[RUN STATUS]
Process        : fhn_train
Job ID         : 4837219
Mode           : train
Start (local)  : 2026-03-01 09:12:44 EST
Nodes          : 2 (node01,node02)
GPUs / node    : 4
CPUs / node    : 64
Status         : FAILED

[RESOURCES BY NODE]
node01: CPU 73% | RAM 205/240 GiB (85%) | GPU util [84%,78%,80%,82%] | VRAM [54/80,50/80,52/80,54/80 GiB]
node02: CPU 61% | RAM 191/240 GiB (80%) | GPU util [72%,75%,70%,73%] | VRAM [46/80,50/80,46/80,49/80 GiB]

[ERROR]
- code: CUDA_OOM
- node: node02
- gpu: 1
- step: 14220
- message: allocation failed in forward pass
```

## Option 6: Four Usage Examples

### Example A: Healthy Training, 2 Nodes

```text
[RUN STATUS]
Process        : fhn_train
Job ID         : 5901123
Mode           : train
Start (local)  : 2026-03-01 10:05:11 EST
Nodes          : 2
GPUs / node    : 4
CPUs / node    : 64
Status         : RUNNING

[CLUSTER CURRENT]
CPU avg        : 67%
RAM            : 402/480 GiB (84%)
GPU util avg   : 81%
VRAM avg       : 418/640 GiB (65%)

[PER NODE]
node01         : CPU 70% | RAM 208/240 GiB (87%) | GPU util avg 84% [83%,86%,82%,85%] | VRAM 214/320 GiB (67%)
node02         : CPU 64% | RAM 194/240 GiB (81%) | GPU util avg 78% [76%,79%,77%,80%] | VRAM 204/320 GiB (64%)

[ERROR]
- none
```

### Example B: Smoke Test, 1 Node

```text
[RUN STATUS]
Process        : fhn_smoke
Job ID         : 5901188
Mode           : smoke
Start (local)  : 2026-03-01 11:20:03 EST
Nodes          : 1
GPUs / node    : 2
CPUs / node    : 32
Status         : RUNNING

[CLUSTER CURRENT]
CPU avg        : 41%
RAM            : 62/128 GiB (48%)
GPU util avg   : 54%
VRAM avg       : 42/160 GiB (26%)

[PER NODE]
node09         : CPU 41% | RAM 62/128 GiB (48%) | GPU util avg 54% [51%,57%] | VRAM 42/160 GiB (26%)

[ERROR]
- none
```

### Example C: CPU-Bound Phase

```text
[RUN STATUS]
Process        : fhn_train
Job ID         : 5901202
Mode           : train
Start (local)  : 2026-03-01 12:02:33 EST
Nodes          : 2
GPUs / node    : 4
CPUs / node    : 64
Status         : RUNNING

[CLUSTER CURRENT]
CPU avg        : 92%
RAM            : 430/480 GiB (90%)
GPU util avg   : 38%
VRAM avg       : 352/640 GiB (55%)

[PER NODE]
node03         : CPU 94% | RAM 220/240 GiB (92%) | GPU util avg 36% [33%,39%,35%,37%] | VRAM 176/320 GiB (55%)
node04         : CPU 90% | RAM 210/240 GiB (88%) | GPU util avg 40% [38%,41%,39%,42%] | VRAM 176/320 GiB (55%)

[ERROR]
- none
```

### Example D: Failed Job (OOM)

```text
[RUN STATUS]
Process        : fhn_train
Job ID         : 5901299
Mode           : train
Start (local)  : 2026-03-01 13:44:18 EST
Nodes          : 2
GPUs / node    : 4
CPUs / node    : 64
Status         : FAILED

[CLUSTER CURRENT]
CPU avg        : 73%
RAM            : 448/480 GiB (93%)
GPU util avg   : 69%
VRAM avg       : 596/640 GiB (93%)

[PER NODE]
node07         : CPU 76% | RAM 228/240 GiB (95%) | GPU util avg 72% [70%,74%,71%,73%] | VRAM 308/320 GiB (96%)
node08         : CPU 70% | RAM 220/240 GiB (92%) | GPU util avg 66% [64%,68%,65%,67%] | VRAM 288/320 GiB (90%)

[ERROR]
- code: CUDA_OOM
- node: node07
- gpu: 2
- step: 18430
- message: allocation failed in forward pass
```
