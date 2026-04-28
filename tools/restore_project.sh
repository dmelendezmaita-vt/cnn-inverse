#!/usr/bin/env bash
set -euo pipefail

SCRIPT_VERSION="1.1.2"

DEFAULT_TARGET="/projects/neuro-collab"
DEFAULT_BACKUP_DEST="/projects/neuro-collab/archives/fhn_dnn"
DEFAULT_BACKUP_CURRENT="prompt"
DEFAULT_FORMAT="tar"

ARCHIVE=""
TARGET="${DEFAULT_TARGET}"
BACKUP_DEST="${DEFAULT_BACKUP_DEST}"
BACKUP_CURRENT="${DEFAULT_BACKUP_CURRENT}"
FORMAT="${DEFAULT_FORMAT}"
YES=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  bash tools/restore_project.sh --archive PATH [OPTIONS]

Description:
  Restore project files from a backup archive and verify integrity before and
  after restore using checksums.

Required:
  --archive PATH         Backup archive to restore from (.tar or .tar.gz).

Options:
  --target PATH          Restore destination root.
                         Default: /projects/neuro-collab
  --backup-current MODE  prompt|always|never
                         prompt (default): ask whether to save current state first.
                         always: always create pre-restore backup.
                         never: never create pre-restore backup.
  --backup-dest PATH     Where pre-restore backup is stored.
                         Default: /projects/neuro-collab/archives/fhn_dnn
  --format tar|tar.gz    Format for pre-restore backup archive.
                         Default: tar
  --yes                  Non-interactive mode.
                         If --backup-current=prompt, this is treated as "yes".
  --dry-run              Validate archive and checksum workflows only.
  -h, --help             Show this help and exit.

Integrity workflow:
  1) Verify <archive>.sha256 against archive bytes.
  2) Extract archive to staging.
  3) Verify staging files against <archive>.files.sha256.
  4) Restore staged files into target.
  5) Verify restored target files against <archive>.files.sha256.

Examples:
  bash tools/restore_project.sh --archive /projects/neuro-collab/archives/fhn_dnn/fhn_dnn_backup_20260228_120000.tar
  bash tools/restore_project.sh --archive /path/to/backup.tar --dry-run
  bash tools/restore_project.sh --archive /path/to/backup.tar --backup-current always --yes
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

ensure_tty_for_prompt() {
  [[ -t 0 ]] || die "Interactive prompt requested but stdin is not a TTY. Re-run with --yes or a non-prompt mode."
}

confirm_default_yes() {
  local prompt="$1"
  local reply
  ensure_tty_for_prompt
  read -r -p "${prompt} [Y/n]: " reply
  case "${reply}" in
    ""|Y|y|yes|YES)
      return 0
      ;;
    N|n|no|NO)
      return 1
      ;;
    *)
      return 1
      ;;
  esac
}

require_overwrite_confirmation() {
  local reply
  ensure_tty_for_prompt
  read -r -p "No pre-restore backup selected. Type OVERWRITE to continue: " reply
  [[ "${reply}" == "OVERWRITE" ]] || die "Restore cancelled by user."
}

verify_archive_checksum() {
  local archive_path="$1"
  local checksum_path="$2"
  local archive_dir archive_name checksum_name

  [[ -f "${checksum_path}" ]] || die "Required archive checksum file not found: ${checksum_path}"

  archive_dir="$(dirname "${archive_path}")"
  archive_name="$(basename "${archive_path}")"
  checksum_name="$(basename "${checksum_path}")"

  log "validating archive checksum: ${checksum_path}"
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
    die "Neither sha256sum nor shasum is available for archive checksum verification."
  fi

  if ! grep -q "${archive_name}" "${checksum_path}"; then
    die "Archive checksum file does not reference archive name: ${archive_name}"
  fi

  log "archive checksum validation passed"
}

verify_files_checksum_manifest() {
  local checksum_file="$1"
  local base_dir="$2"
  local count=0
  local line expected rel actual

  [[ -f "${checksum_file}" ]] || die "Required file checksum manifest not found: ${checksum_file}"

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

extract_archive_to_dir() {
  local archive_path="$1"
  local out_dir="$2"

  mkdir -p "${out_dir}"

  if [[ "${archive_path}" == *.tar.gz || "${archive_path}" == *.tgz ]]; then
    tar -xzf "${archive_path}" -C "${out_dir}"
  else
    tar -xf "${archive_path}" -C "${out_dir}"
  fi
}

create_pre_restore_backup() {
  local script_dir backup_script

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  backup_script="${script_dir}/backup_project.sh"
  [[ -f "${backup_script}" ]] || die "backup script not found: ${backup_script}"

  log "creating pre-restore backup from target: ${TARGET} (scope=repo)"
  bash "${backup_script}" \
    --source "${TARGET}" \
    --dest "${BACKUP_DEST}" \
    --prefix "fhn_dnn_pre_restore" \
    --format "${FORMAT}" \
    --scope repo
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive)
      [[ $# -ge 2 ]] || die "Missing value for --archive"
      ARCHIVE="$2"
      shift 2
      ;;
    --target)
      [[ $# -ge 2 ]] || die "Missing value for --target"
      TARGET="$2"
      shift 2
      ;;
    --backup-current)
      [[ $# -ge 2 ]] || die "Missing value for --backup-current"
      BACKUP_CURRENT="$2"
      shift 2
      ;;
    --backup-dest)
      [[ $# -ge 2 ]] || die "Missing value for --backup-dest"
      BACKUP_DEST="$2"
      shift 2
      ;;
    --format)
      [[ $# -ge 2 ]] || die "Missing value for --format"
      FORMAT="$2"
      shift 2
      ;;
    --yes)
      YES=1
      shift
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

[[ -n "${ARCHIVE}" ]] || die "--archive is required. Use --help for usage."
[[ -f "${ARCHIVE}" ]] || die "Archive not found: ${ARCHIVE}"
[[ "${BACKUP_CURRENT}" == "prompt" || "${BACKUP_CURRENT}" == "always" || "${BACKUP_CURRENT}" == "never" ]] || die "--backup-current must be prompt, always, or never"
[[ "${FORMAT}" == "tar" || "${FORMAT}" == "tar.gz" ]] || die "--format must be tar or tar.gz"
[[ -n "${TARGET}" ]] || die "--target cannot be empty"
[[ "${TARGET}" != "/" ]] || die "Refusing to restore into '/'"

ARCHIVE_CHECKSUM_PATH="${ARCHIVE}.sha256"
FILES_CHECKSUM_PATH="${ARCHIVE}.files.sha256"

log "restore configuration"
log "  archive=${ARCHIVE}"
log "  target=${TARGET}"
log "  backup_current=${BACKUP_CURRENT}"
log "  backup_dest=${BACKUP_DEST}"
log "  format=${FORMAT}"
log "  yes=${YES}"
log "  dry_run=${DRY_RUN}"

verify_archive_checksum "${ARCHIVE}" "${ARCHIVE_CHECKSUM_PATH}"

STAGING_DIR="$(mktemp -d)"
cleanup() {
  if [[ -n "${STAGING_DIR:-}" && -d "${STAGING_DIR}" ]]; then
    rm -rf "${STAGING_DIR}"
  fi
}
trap cleanup EXIT

log "extracting archive to staging: ${STAGING_DIR}"
extract_archive_to_dir "${ARCHIVE}" "${STAGING_DIR}"

log "verifying staging files checksum manifest: ${FILES_CHECKSUM_PATH}"
verify_files_checksum_manifest "${FILES_CHECKSUM_PATH}" "${STAGING_DIR}"
log "staging file checksum verification passed"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  case "${BACKUP_CURRENT}" in
    always)
      log "dry-run: would create pre-restore backup"
      ;;
    never)
      log "dry-run: would skip pre-restore backup"
      ;;
    prompt)
      if [[ "${YES}" -eq 1 ]]; then
        log "dry-run: prompt mode with --yes; would create pre-restore backup"
      else
        log "dry-run: prompt mode without --yes; interactive prompt would decide pre-restore backup"
      fi
      ;;
  esac
  log "dry-run: would copy staged files into ${TARGET}"
  log "dry-run: would verify restored files checksum in ${TARGET}"
  log "dry-run completed; no files changed"
  exit 0
fi

create_backup=0
case "${BACKUP_CURRENT}" in
  always)
    create_backup=1
    ;;
  never)
    create_backup=0
    ;;
  prompt)
    if [[ "${YES}" -eq 1 ]]; then
      create_backup=1
    else
      if confirm_default_yes "Create pre-restore backup of current target before restoring?"; then
        create_backup=1
      else
        create_backup=0
      fi
    fi
    ;;
esac

if [[ "${create_backup}" -eq 1 ]]; then
  create_pre_restore_backup
else
  if [[ "${YES}" -eq 0 ]]; then
    require_overwrite_confirmation
  fi
fi

mkdir -p "${TARGET}"
log "restoring staged files into target"
if has_cmd rsync; then
  rsync -rlt --omit-dir-times --no-perms --no-owner --no-group "${STAGING_DIR}/" "${TARGET}/"
else
  (cd "${STAGING_DIR}" && tar -cf - .) | (cd "${TARGET}" && tar -xmf - --no-overwrite-dir)
fi

log "verifying restored files checksum in target"
verify_files_checksum_manifest "${FILES_CHECKSUM_PATH}" "${TARGET}"
log "restore completed successfully with post-restore checksum verification"
