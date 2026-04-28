# GitHub Push Checklist

## Current Local State

| Item | Value |
| --- | --- |
| Local staging repo path | `/projects/neuro-collab/code/github_prep/fhn_dnn-1-implementation-in-pytorch-public` |
| Default branch | `main` |
| Current staging commit | `8ea0b6228dbc132417eb17e715398b05b83cc9b8` |
| Remote | `git@github-dmelendezmaita-vt:dmelendezmaita-vt/fhn_dnn-1-implementation-in-pytorch.git` |
| Suggested owner namespace from local SSH config | `dmelendezmaita-vt` |

## Commands

The local SSH configuration already defines a GitHub host alias for `dmelendezmaita-vt`, so that namespace is the best-supported default unless a different owner is intended.

```bash
cd /projects/neuro-collab/code/github_prep/fhn_dnn-1-implementation-in-pytorch-public
git push -u origin main
```

For HTTPS instead of SSH:

```bash
cd /projects/neuro-collab/code/github_prep/fhn_dnn-1-implementation-in-pytorch-public
git remote set-url origin https://github.com/dmelendezmaita-vt/fhn_dnn-1-implementation-in-pytorch.git
git push -u origin main
```

## Final Pre-Push Checks

| Check | Command |
| --- | --- |
| Confirm remote | `git remote -v` |
| Confirm clean working tree | `git status --short` |
| Reconfirm no GitHub-oversize files | `find . -type f -size +95M` |
| Reconfirm staged scope docs | `sed -n '1,200p' README.md` |
