# GitHub Upload Scope

This document records the exact scope of the public GitHub staging tree.

## Source Condition

| Item | Observed state | Consequence |
| --- | --- | --- |
| Live project tree | Unpacked working copy under `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch` | The active source tree has no attached `.git` directory |
| Archive history | Zip and tar backups exist locally | File recovery is possible, but historical git branches and remotes are not recoverable from the backup format alone |
| Public remote | Local SSH rewrite exists for `dmelendezmaita-vt` | The staging repo is prepared locally and can target that owner namespace unless a different destination is chosen |
| Zip source markers | Both code archives carry 40-hex zip comments | They likely record source snapshot identifiers, even though no local git graph survived the restore path |

## Included Scope

| Area | Included | Reason |
| --- | --- | --- |
| Core source directories | Yes | Required for executable reproduction of the baseline code paths |
| `third_party/dl-kit-main/` | Yes | Required to satisfy the `dlkit` imports used by the PyTorch workflow |
| Baseline FHN sample data | Yes | Small enough for GitHub and needed for a real public run path |
| Curated text-first evidence folders | Yes | Needed to preserve the canonical reporting chain without shipping runtime noise |
| Prior upload bundle | No | The older allowlist-based packaging bundle is intentionally excluded so the current public tree stays aligned with the canonical reproduction chain |
| `repro/` | Yes | Stable human entrypoint for the canonical experiment chain |

## Excluded Scope

| Excluded path class | Reason |
| --- | --- |
| `sbi-logs/`, live run directories, and scratch mirrors | Runtime outputs rather than source artifacts |
| `data/important_notes/**/logs/` and queue `.out` / `.err` files | Operational noise, often large, with little reuse value in a public repo |
| Binary intermediate arrays under note folders | GitHub size hygiene and reproducibility focus on manifests and reports |
| Tool-local files such as `WARP.md`, local virtual environments, and `__pycache__` trees | Not part of the public scientific artifact |
| Full paper-template vendoring under `docs/paper/acm_template/` | Third-party template bulk that does not advance reproducibility of the code |

## Remaining Host-Specific Interfaces

| Interface | Current state | Recommended next step |
| --- | --- | --- |
| SBATCH launchers | Preserve ARC-specific absolute paths | Either keep them as reference artifacts or add wrapper launchers that derive paths from the repo root |
| Python builder and launcher scripts | Many still encode `/projects/neuro-collab/...` | Introduce a small path-configuration layer if the public repo will be executed outside the original cluster |
| Full Hodgkin-Huxley data access | External only | Publish a reconstruction guide or a dataset handoff recipe if redistribution is allowed |

## Immediate GitHub Steps

1. Review the `repro/` layer first, because it is now the intended public reading order.
2. Review any remaining local-only paths that should be documented rather than executed directly.
3. Push the staged `main` branch and, if needed, add release notes that explain the excluded runtime artifacts and the untouched upstream archive reference.
