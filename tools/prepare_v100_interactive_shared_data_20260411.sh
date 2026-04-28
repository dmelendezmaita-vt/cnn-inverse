#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${REPO}/data/important_notes/optimization_track_20260411_v100_interactive/shared_data"
NOTES="${REPO}/data/important_notes/optimization_track_20260411_v100_interactive/notes"

mkdir -p "${ROOT}" "${NOTES}"

extract_one() {
  local tar_path="$1"
  local out_dir="$2"
  local tmp_dir

  if [[ -d "${out_dir}" ]] && find "${out_dir}" -mindepth 1 -maxdepth 1 | grep -q .; then
    echo "$(date -Is) shared-data exists: ${out_dir}"
    return 0
  fi

  rm -rf "${out_dir}.tmp"
  mkdir -p "${out_dir}.tmp"
  echo "$(date -Is) extracting ${tar_path} -> ${out_dir}.tmp"
  tar -xf "${tar_path}" -C "${out_dir}.tmp"

  local first_dir=""
  first_dir="$(find "${out_dir}.tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
  rm -rf "${out_dir}"
  if [[ -n "${first_dir}" ]]; then
    mv "${first_dir}" "${out_dir}"
    rm -rf "${out_dir}.tmp"
  else
    mv "${out_dir}.tmp" "${out_dir}"
  fi

  echo "$(date -Is) shared-data ready: ${out_dir}"
}

extract_one \
  "/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar" \
  "${ROOT}/track3_hh_reduced/reduced_data"

extract_one \
  "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar" \
  "${ROOT}/track4_hh_full/concatenated_data"

cat > "${NOTES}/shared_data_prep_20260411.txt" <<EOF
timestamp=$(date -Is)
track3_shared_dir=${ROOT}/track3_hh_reduced/reduced_data
track4_shared_dir=${ROOT}/track4_hh_full/concatenated_data
EOF

echo "$(date -Is) shared-data preparation complete"
