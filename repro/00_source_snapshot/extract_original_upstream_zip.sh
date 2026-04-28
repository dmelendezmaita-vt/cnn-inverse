#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-/projects/neuro-collab/code/archives/fhn_dnn-1-implementation-in-pytorch.zip}"
DEST_ROOT="${2:-./upstream_original}"
EXPECTED_SIZE="110818160"
EXPECTED_COMMENT="eb676a34bb32d880b172e70f9faf6f41a2d9fe3c"

[[ -f "${ZIP_PATH}" ]] || { echo "ERROR: archive not found: ${ZIP_PATH}" >&2; exit 1; }
command -v zipinfo >/dev/null 2>&1 || { echo "ERROR: zipinfo is required." >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "ERROR: unzip is required." >&2; exit 1; }

ACTUAL_SIZE="$(stat -c '%s' "${ZIP_PATH}")"
[[ "${ACTUAL_SIZE}" == "${EXPECTED_SIZE}" ]] || { echo "ERROR: unexpected archive size: ${ACTUAL_SIZE}" >&2; exit 1; }

ACTUAL_COMMENT="$(zipinfo -z "${ZIP_PATH}" | head -n 2 | tail -n 1 | tr -d '\r')"
[[ "${ACTUAL_COMMENT}" == "${EXPECTED_COMMENT}" ]] || { echo "ERROR: unexpected zip comment: ${ACTUAL_COMMENT}" >&2; exit 1; }

mkdir -p "${DEST_ROOT}"
unzip -q "${ZIP_PATH}" -d "${DEST_ROOT}"
echo "Extracted ${ZIP_PATH} to ${DEST_ROOT}"
