#!/usr/bin/env bash
set -euo pipefail

SCRIPT_VERSION="1.1.1"

DEFAULT_SOURCE="/projects/neuro-collab"
DEFAULT_DEST="/projects/neuro-collab/archives/fhn_dnn"
DEFAULT_PREFIX="fhn_dnn_backup"
DEFAULT_FORMAT="tar"
DEFAULT_SCOPE="key"

SOURCE="${DEFAULT_SOURCE}"
DEST="${DEFAULT_DEST}"
PREFIX="${DEFAULT_PREFIX}"
FORMAT="${DEFAULT_FORMAT}"
SCOPE="${DEFAULT_SCOPE}"
DRY_RUN=0

KEY_INCLUDE_PATHS=(
  "code/fhn_dnn-1-implementation-in-pytorch/pytorch"
  "code/fhn_dnn-1-implementation-in-pytorch/tensorflow"
  "code/fhn_dnn-1-implementation-in-pytorch/utils"
  "code/fhn_dnn-1-implementation-in-pytorch/scripts"
  "code/fhn_dnn-1-implementation-in-pytorch/docs"
  "code/fhn_dnn-1-implementation-in-pytorch/slurm_train.sbatch"
  "code/fhn_dnn-1-implementation-in-pytorch/slurm_smoke_test.sbatch"
  "code/fhn_dnn-1-implementation-in-pytorch/README.md"
  "code/fhn_dnn-1-implementation-in-pytorch/WARP.md"
  "code/fhn_dnn-1-implementation-in-pytorch/LICENSE"
  "code/dl-kit-main/dlkit/opt/train.py"
  "code/dl-kit-main/dlkit/opt/train_utils.py"
)

CACHE_EXCLUDES=(
  "*/__pycache__/*"
  "*.pyc"
)

ALWAYS_EXCLUDES=(
  "share/*"
)

REPO_EXCLUDES=(
  "archives/*"
  "data/*"
  "runs/*"
  "other/*"
  ".codex/*"
  "code/fhn_dnn-1-implementation-in-pytorch/data/*"
  "code/fhn_dnn-1-implementation-in-pytorch/.git/*"
)

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/backup_project.sh [OPTIONS]

Description:
  Create a timestamped project backup archive with integrity sidecars:
  - <archive>.sha256                 (archive checksum)
  - <archive>.files.sha256           (file-level checksums for backup content)
  - <archive>.filelist.txt           (exact relative file list in archive)
  - <archive>.manifest.txt           (metadata)

Defaults:
  --source /projects/neuro-collab
  --dest   /projects/neuro-collab/archives/fhn_dnn
  --prefix fhn_dnn_backup
  --format tar
  --scope  key

Options:
  --source PATH          Project root to back up.
  --dest PATH            Output directory for archive and sidecars.
  --prefix NAME          Archive filename prefix.
  --format tar|tar.gz    Archive format. Default: tar.
  --scope key|repo       key: curated scope, repo: full source minus excludes.
  --dry-run              Print planned actions, do not create files.
  -h, --help             Show this help and exit.

Integrity guarantees at backup time:
  1) Generate file-level checksums from source files.
  2) Create archive.
  3) Verify archive checksum sidecar.
  4) Extract archive to temp and verify:
     - extracted file list exactly matches source file list
     - extracted file checksums exactly match source file checksums

Examples:
  bash scripts/backup_project.sh
  bash scripts/backup_project.sh --scope repo --format tar.gz
  bash scripts/backup_project.sh --dry-run
USAGE
}

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

hash_file() {
  local file_path="$1"
  if has_cmd sha256sum; then
    sha256sum "${file_path}" | awk '{print $1}'
  elif has_cmd shasum; then
    shasum -a 256 "${file_path}" | awk '{print $1}'
  else
    die "Neither sha256sum nor shasum is available on PATH."
  fi
}

verify_archive_checksum() {
  local archive_path="$1"
  local checksum_path="$2"
  local archive_dir archive_name checksum_name

  archive_dir="$(dirname "${archive_path}")"
  archive_name="$(basename "${archive_path}")"
  checksum_name="$(basename "${checksum_path}")"

  if has_cmd sha256sum; then
    (
      cd "${archive_dir}"
      sha256sum -c "${checksum_name}" >/dev/null
    )
  elif has_cmd shasum; then
    (
      cd "${archive_dir}"
      shasum -a 256 -c "${checksum_name}" >/dev/null
    )
  else
    die "Neither sha256sum nor shasum is available for checksum verification."
  fi

  if ! grep -q "${archive_name}" "${checksum_path}"; then
    die "Checksum file exists but does not reference archive name: ${archive_name}"
  fi
}

write_archive_checksum() {
  local archive_path="$1"
  local checksum_path="$2"
  local archive_dir archive_name checksum_name

  archive_dir="$(dirname "${archive_path}")"
  archive_name="$(basename "${archive_path}")"
  checksum_name="$(basename "${checksum_path}")"

  if has_cmd sha256sum; then
    (
      cd "${archive_dir}"
      sha256sum "${archive_name}" > "${checksum_name}"
    )
  elif has_cmd shasum; then
    (
      cd "${archive_dir}"
      shasum -a 256 "${archive_name}" > "${checksum_name}"
    )
  else
    die "Neither sha256sum nor shasum is available on PATH."
  fi
}

should_exclude() {
  local rel="$1"
  shift
  local pattern
  for pattern in "$@"; do
    if [[ "${rel}" == ${pattern} ]]; then
      return 0
    fi
  done
  return 1
}

generate_file_list() {
  local out_file="$1"
  local raw_file
  raw_file="$(mktemp)"

  if [[ "${SCOPE}" == "key" ]]; then
    local p
    for p in "${KEY_INCLUDE_PATHS[@]}"; do
      if [[ -d "${SOURCE}/${p}" ]]; then
        find "${SOURCE}/${p}" -type f | sed "s|^${SOURCE}/||" >> "${raw_file}"
      elif [[ -f "${SOURCE}/${p}" ]]; then
        printf '%s\n' "${p}" >> "${raw_file}"
      else
        rm -f "${raw_file}"
        die "Required key-scope path missing from source: ${p}"
      fi
    done
  else
    find "${SOURCE}" -type f | sed "s|^${SOURCE}/||" > "${raw_file}"
  fi

  LC_ALL=C sort -u "${raw_file}" > "${raw_file}.sorted"

  : > "${out_file}"
  local rel
  while IFS= read -r rel; do
    [[ -n "${rel}" ]] || continue
    if [[ "${SCOPE}" == "key" ]]; then
      if should_exclude "${rel}" "${CACHE_EXCLUDES[@]}" "${ALWAYS_EXCLUDES[@]}"; then
        continue
      fi
    else
      if should_exclude "${rel}" "${CACHE_EXCLUDES[@]}" "${ALWAYS_EXCLUDES[@]}" "${REPO_EXCLUDES[@]}"; then
        continue
      fi
    fi
    printf '%s\n' "${rel}" >> "${out_file}"
  done < "${raw_file}.sorted"

  rm -f "${raw_file}" "${raw_file}.sorted"

  [[ -s "${out_file}" ]] || die "File list is empty after applying scope and excludes."
}

write_files_checksum_manifest() {
  local manifest_file="$1"
  local file_list_file="$2"
  local base_dir="$3"

  : > "${manifest_file}"
  local rel hash
  while IFS= read -r rel; do
    [[ -n "${rel}" ]] || continue
    [[ -f "${base_dir}/${rel}" ]] || die "Missing source file while hashing: ${rel}"
    hash="$(hash_file "${base_dir}/${rel}")"
    printf '%s  %s\n' "${hash}" "${rel}" >> "${manifest_file}"
  done < "${file_list_file}"
}

verify_files_checksum_manifest() {
  local checksum_file="$1"
  local base_dir="$2"
  local count=0
  local line expected rel actual

  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -n "${line}" ]] || continue
    expected="${line%% *}"
    rel="${line#*  }"
    [[ -n "${rel}" ]] || die "Invalid checksum line in ${checksum_file}: ${line}"
    [[ -f "${base_dir}/${rel}" ]] || die "Missing file for checksum verification: ${base_dir}/${rel}"
    actual="$(hash_file "${base_dir}/${rel}")"
    [[ "${actual}" == "${expected}" ]] || die "Checksum mismatch for ${rel}"
    count=$((count + 1))
  done < "${checksum_file}"

  [[ ${count} -gt 0 ]] || die "No checksum entries found in ${checksum_file}"
}

verify_exact_copy_from_archive() {
  local archive_path="$1"
  local file_list_file="$2"
  local files_checksum_path="$3"
  local tmp_dir extracted_dir extracted_list

  tmp_dir="$(mktemp -d)"
  extracted_dir="${tmp_dir}/extract"
  extracted_list="${tmp_dir}/extracted_files.txt"
  mkdir -p "${extracted_dir}"

  if [[ "${archive_path}" == *.tar.gz || "${archive_path}" == *.tgz ]]; then
    tar -xzf "${archive_path}" -C "${extracted_dir}"
  else
    tar -xf "${archive_path}" -C "${extracted_dir}"
  fi

  find "${extracted_dir}" -type f | sed "s|^${extracted_dir}/||" | LC_ALL=C sort > "${extracted_list}"

  if ! diff -u "${file_list_file}" "${extracted_list}" >/dev/null; then
    rm -rf "${tmp_dir}"
    die "Archive content list differs from source file list."
  fi

  verify_files_checksum_manifest "${files_checksum_path}" "${extracted_dir}"
  rm -rf "${tmp_dir}"
}

write_manifest() {
  local manifest_path="$1"
  local archive_path="$2"
  local timestamp="$3"
  local archive_checksum_path="$4"
  local files_checksum_path="$5"
  local file_list_path="$6"
  local file_count="$7"

  {
    echo "backup_script_version=${SCRIPT_VERSION}"
    echo "created_at=${timestamp}"
    echo "created_at_iso=$(date -Is)"
    echo "host=$(hostname)"
    echo "user=${USER:-unknown}"
    echo "source=${SOURCE}"
    echo "dest=${DEST}"
    echo "prefix=${PREFIX}"
    echo "format=${FORMAT}"
    echo "scope=${SCOPE}"
    echo "archive_path=${archive_path}"
    echo "archive_checksum_path=${archive_checksum_path}"
    echo "files_checksum_path=${files_checksum_path}"
    echo "file_list_path=${file_list_path}"
    echo "file_count=${file_count}"
    echo
    if [[ "${SCOPE}" == "key" ]]; then
      echo "[included_paths]"
      local p
      for p in "${KEY_INCLUDE_PATHS[@]}"; do
        echo "${p}"
      done
      echo
      echo "[excluded_patterns]"
      for p in "${CACHE_EXCLUDES[@]}" "${ALWAYS_EXCLUDES[@]}"; do
        echo "${p}"
      done
    else
      echo "[excluded_patterns]"
      for p in "${CACHE_EXCLUDES[@]}" "${ALWAYS_EXCLUDES[@]}" "${REPO_EXCLUDES[@]}"; do
        echo "${p}"
      done
    fi
  } > "${manifest_path}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 ]] || die "Missing value for --source"
      SOURCE="$2"
      shift 2
      ;;
    --dest)
      [[ $# -ge 2 ]] || die "Missing value for --dest"
      DEST="$2"
      shift 2
      ;;
    --prefix)
      [[ $# -ge 2 ]] || die "Missing value for --prefix"
      PREFIX="$2"
      shift 2
      ;;
    --format)
      [[ $# -ge 2 ]] || die "Missing value for --format"
      FORMAT="$2"
      shift 2
      ;;
    --scope)
      [[ $# -ge 2 ]] || die "Missing value for --scope"
      SCOPE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ -d "${SOURCE}" ]] || die "Source directory does not exist: ${SOURCE}"
[[ "${FORMAT}" == "tar" || "${FORMAT}" == "tar.gz" ]] || die "--format must be tar or tar.gz"
[[ "${SCOPE}" == "key" || "${SCOPE}" == "repo" ]] || die "--scope must be key or repo"

file_list_tmp="$(mktemp)"
generate_file_list "${file_list_tmp}"
file_count="$(wc -l < "${file_list_tmp}" | awk '{print $1}')"

ts="$(date +"%Y%m%d_%H%M%S")"
ext="tar"
if [[ "${FORMAT}" == "tar.gz" ]]; then
  ext="tar.gz"
fi

archive_path="${DEST}/${PREFIX}_${ts}.${ext}"
archive_checksum_path="${archive_path}.sha256"
files_checksum_path="${archive_path}.files.sha256"
file_list_path="${archive_path}.filelist.txt"
manifest_path="${archive_path}.manifest.txt"

log "backup configuration"
log "  source=${SOURCE}"
log "  dest=${DEST}"
log "  prefix=${PREFIX}"
log "  format=${FORMAT}"
log "  scope=${SCOPE}"
log "  dry_run=${DRY_RUN}"
log "  file_count=${file_count}"
log "  archive=${archive_path}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "dry-run file list preview:"
  sed -n '1,80p' "${file_list_tmp}" | while IFS= read -r line; do
    log "  - ${line}"
  done
  rm -f "${file_list_tmp}"
  log "dry-run completed; no files created"
  exit 0
fi

mkdir -p "${DEST}"

if [[ -e "${archive_path}" || -e "${archive_checksum_path}" || -e "${files_checksum_path}" || -e "${file_list_path}" || -e "${manifest_path}" ]]; then
  rm -f "${file_list_tmp}"
  die "Refusing to overwrite existing output. Re-run after 1 second or choose a different prefix."
fi

cp "${file_list_tmp}" "${file_list_path}"
write_files_checksum_manifest "${files_checksum_path}" "${file_list_tmp}" "${SOURCE}"

if [[ "${FORMAT}" == "tar" ]]; then
  tar -cf "${archive_path}" -C "${SOURCE}" -T "${file_list_tmp}"
else
  tar -czf "${archive_path}" -C "${SOURCE}" -T "${file_list_tmp}"
fi

write_archive_checksum "${archive_path}" "${archive_checksum_path}"
verify_archive_checksum "${archive_path}" "${archive_checksum_path}"
verify_exact_copy_from_archive "${archive_path}" "${file_list_tmp}" "${files_checksum_path}"

write_manifest \
  "${manifest_path}" \
  "${archive_path}" \
  "${ts}" \
  "${archive_checksum_path}" \
  "${files_checksum_path}" \
  "${file_list_path}" \
  "${file_count}"

rm -f "${file_list_tmp}"

if has_cmd du; then
  archive_size="$(du -h "${archive_path}" | awk '{print $1}')"
  log "archive created: ${archive_path} (${archive_size})"
else
  log "archive created: ${archive_path}"
fi
log "archive checksum: ${archive_checksum_path}"
log "files checksum:   ${files_checksum_path}"
log "file list:        ${file_list_path}"
log "manifest:         ${manifest_path}"
log "backup integrity verification passed"
