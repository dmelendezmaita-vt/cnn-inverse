# Source Snapshot Recovery

The inherited baseline depends on an untouched upstream archive that is not stored directly in this GitHub staging tree, because the zip is larger than GitHub's single-file hard limit. Reproducibility therefore requires an exact external reference to that archive.

## Exact Archive Reference

| Field | Value |
| --- | --- |
| archive filename | `fhn_dnn-1-implementation-in-pytorch.zip` |
| local archive path in the original workspace | `/projects/neuro-collab/code/archives/fhn_dnn-1-implementation-in-pytorch.zip` |
| zip size | `110818160` bytes |
| zip comment | `eb676a34bb32d880b172e70f9faf6f41a2d9fe3c` |
| top-level extracted directory | `fhn_dnn-1-implementation-in-pytorch/` |

## Exact Recovery Commands

Verify the marker:

```bash
unzip -z /projects/neuro-collab/code/archives/fhn_dnn-1-implementation-in-pytorch.zip
```

Recover the untouched source tree:

```bash
tmp="$(mktemp -d /tmp/fhn_zip_baseline.XXXXXX)" && unzip -q /projects/neuro-collab/code/archives/fhn_dnn-1-implementation-in-pytorch.zip -d "$tmp"
```

Or use the helper:

```bash
bash repro/00_source_snapshot/extract_original_upstream_zip.sh \
  /projects/neuro-collab/code/archives/fhn_dnn-1-implementation-in-pytorch.zip \
  ./upstream_original
```

## Closest In-Repo Mirrors

| Path | Role |
| --- | --- |
| `evidence/canonical_chain/10_fhn_benchmark_reconstruction/first_track_paper_parity/fhn_dnn_original_unmodified_20260302_050951/` | extracted benchmark-era reference bundle used during the first-track parity work |
| `evidence/canonical_chain/10_fhn_benchmark_reconstruction/first_track_paper_parity/code_changes_vs_code_archives_zip_20260302.md` | explicit local provenance note for the untouched archive reference |
