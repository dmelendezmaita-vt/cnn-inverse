#!/usr/bin/env bash
set -euo pipefail

CSV="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/v100_paper_data_parity_training_jobs.csv"
TMP_DIR="/tmp/v100_equiv_update"
mkdir -p "$TMP_DIR"

# collect ids from CSV (column 15)
ids="$(awk -F',' 'NR>1{print $15}' "$CSV" | paste -sd, -)"
if [[ -z "$ids" ]]; then
  echo "No job IDs in $CSV"
  exit 0
fi

SQUEUE_MAP="$TMP_DIR/squeue_map.txt"
SACCT_MAP="$TMP_DIR/sacct_map.txt"

ssh -o BatchMode=yes tinkercliffs1 "ssh -o BatchMode=yes falcon1.arc.vt.edu \"squeue -h -j ${ids} -o '%i|%T|%M'\"" > "$SQUEUE_MAP" || true
ssh -o BatchMode=yes tinkercliffs1 "ssh -o BatchMode=yes falcon1.arc.vt.edu \"sacct -j ${ids} --format=JobIDRaw,State,Elapsed -P -n\"" > "$SACCT_MAP" || true

awk -F',' -v SQUEUE_MAP="$SQUEUE_MAP" -v SACCT_MAP="$SACCT_MAP" '
BEGIN {
  OFS=","
  # base map from sacct (terminal states)
  while ((getline line < SACCT_MAP) > 0) {
    split(line,a,"|")
    if (a[1] ~ /^[0-9]+$/) {
      st[a[1]] = a[2]
      el[a[1]] = a[3]
    }
  }
  # override with live squeue status
  while ((getline line < SQUEUE_MAP) > 0) {
    split(line,a,"|")
    st[a[1]] = a[2]
    el[a[1]] = a[3]
  }
}
NR==1 { print; next }
{
  jid = $15
  run_dir = $19
  if (jid in st) {
    $16 = st[jid]
    if ($16 == "PENDING") $17 = "N/A"; else $17 = el[jid]
  }

  loss = run_dir "/loss.txt"
  if ((($16 == "COMPLETED") || ($16 == "FAILED") || ($16 == "CANCELLED") || ($16 ~ /TIMEOUT/)) && system("test -s \"" loss "\"") == 0) {
    cmd1 = "awk '\''END{print $1}'\'' \"" loss "\""
    cmd2 = "awk '\''END{print $2}'\'' \"" loss "\""
    cmd3 = "awk '\''NR==1{m=$1} $1<m{m=$1} END{print m}'\'' \"" loss "\""
    cmd4 = "awk '\''NR==1{m=$2} $2<m{m=$2} END{print m}'\'' \"" loss "\""
    cmd1 | getline train_last; close(cmd1)
    cmd2 | getline val_last; close(cmd2)
    cmd3 | getline train_best; close(cmd3)
    cmd4 | getline val_best; close(cmd4)
    if (train_last != "") $20 = train_last
    if (val_last != "") $21 = val_last
    if (train_best != "") $22 = train_best
    if (val_best != "") $23 = val_best
  }

  print
}
' "$CSV" > "$CSV.tmp"

mv "$CSV.tmp" "$CSV"
rm -rf "$TMP_DIR"

echo "Updated $CSV"
