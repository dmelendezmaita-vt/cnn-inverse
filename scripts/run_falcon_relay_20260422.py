#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

DEFAULT_TC_HOST = "tinkercliffs1"
DEFAULT_FALCON_HOST = "falcon1.arc.vt.edu"
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class StepSpec:
    alloc_job_id: str
    nodelist: Optional[str] = None
    nodes: int = 1
    ntasks: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: Optional[int] = None
    gres: Optional[str] = None
    exact: bool = True
    exclusive: bool = False
    kill_on_bad_exit: int = 1


def parse_env_pairs(raw_pairs: Sequence[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for raw in raw_pairs:
        if "=" not in raw:
            raise SystemExit(f"Expected KEY=VALUE for --env, got: {raw}")
        key, value = raw.split("=", 1)
        if not ENV_KEY_RE.fullmatch(key):
            raise SystemExit(f"Invalid environment variable name: {key}")
        env[key] = value
    return env


def strip_command_prefix(command: Sequence[str]) -> List[str]:
    items = list(command)
    if items and items[0] == "--":
        items = items[1:]
    if not items:
        raise SystemExit("Need a command after --")
    return items


def build_shell_script(command: Sequence[str], *, workdir: Optional[str], env: Dict[str, str]) -> str:
    argv = strip_command_prefix(command)
    steps = ["set -euo pipefail"]
    if workdir:
        steps.append(f"cd {shlex.quote(workdir)}")
    for key, value in env.items():
        steps.append(f"export {key}={shlex.quote(value)}")
    steps.append(shlex.join(argv))
    return " && ".join(steps)


def build_srun_command(step: StepSpec, shell_script: str) -> List[str]:
    if step.exact and step.exclusive:
        raise ValueError("Choose either exact or exclusive step placement, not both.")

    cmd = ["srun", f"--jobid={step.alloc_job_id}"]
    if step.exact:
        cmd.append("--exact")
    if step.exclusive:
        cmd.append("--exclusive")
    if step.nodes > 0:
        cmd.append(f"-N{step.nodes}")
    if step.nodelist:
        cmd.append(f"-w{step.nodelist}")
    if step.ntasks > 0:
        cmd.append(f"--ntasks={step.ntasks}")
    if step.ntasks_per_node > 0:
        cmd.append(f"--ntasks-per-node={step.ntasks_per_node}")
    if step.cpus_per_task is not None:
        cmd.append(f"--cpus-per-task={step.cpus_per_task}")
    if step.gres:
        cmd.append(f"--gres={step.gres}")
    cmd.append(f"--kill-on-bad-exit={step.kill_on_bad_exit}")
    cmd.extend(["bash", "-lc", shell_script])
    return cmd


def build_falcon_relay_command(
    remote_shell_script: str,
    *,
    tc_host: str = DEFAULT_TC_HOST,
    falcon_host: str = DEFAULT_FALCON_HOST,
) -> List[str]:
    inner = f"bash -lc {shlex.quote(remote_shell_script)}"
    outer = f"ssh -n -o BatchMode=yes {falcon_host} {shlex.quote(inner)}"
    return [
        "ssh",
        "-n",
        "-o",
        "BatchMode=yes",
        tc_host,
        outer,
    ]


def render_local_command(cmd: Sequence[str]) -> str:
    return shlex.join(list(cmd))


def run_local_command(cmd: Sequence[str], *, capture_output: bool) -> int:
    if capture_output:
        proc = subprocess.run(list(cmd), text=True, capture_output=True, check=False)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.returncode
    proc = subprocess.run(list(cmd), check=False)
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tc-host", default=DEFAULT_TC_HOST)
    common.add_argument("--falcon-host", default=DEFAULT_FALCON_HOST)
    common.add_argument("--dry-run", action="store_true", help="print the final local relay command and exit")
    common.add_argument(
        "--capture-output",
        action="store_true",
        help="capture output before printing it back locally instead of inheriting stdio",
    )

    parser = argparse.ArgumentParser(
        description="Relay commands from this environment through tinkercliffs1 -> falcon1, with optional srun steps inside a live Falcon allocation.",
        epilog=(
            "Examples:\n"
            "  python run_falcon_relay_20260422.py remote -- squeue -h -j 368960 -o '%i|%T|%N'\n"
            "  python run_falcon_relay_20260422.py step --alloc-job-id 368960 --nodelist fal117 "
            "--cpus-per-task 1 --gres gpu:v100:1 -- nvidia-smi -L"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    remote = subparsers.add_parser("remote", parents=[common], help="run a command on falcon1 through the relay")
    remote.add_argument("--workdir")
    remote.add_argument("--env", action="append", default=[], help="repeatable KEY=VALUE exported on falcon1")
    remote.add_argument("command", nargs=argparse.REMAINDER)

    step = subparsers.add_parser("step", parents=[common], help="run an srun step inside a live Falcon allocation")
    step.add_argument("--alloc-job-id", required=True)
    step.add_argument("--nodelist", help="optional Falcon host or nodelist, for example fal117")
    step.add_argument("--nodes", type=int, default=1)
    step.add_argument("--ntasks", type=int, default=1)
    step.add_argument("--ntasks-per-node", type=int, default=1)
    step.add_argument("--cpus-per-task", type=int)
    step.add_argument("--gres", help="for example gpu:v100:1")
    step.add_argument("--workdir")
    step.add_argument("--env", action="append", default=[], help="repeatable KEY=VALUE exported inside the step")
    exact_group = step.add_mutually_exclusive_group()
    exact_group.add_argument("--exact", dest="exact", action="store_true", default=True)
    exact_group.add_argument("--exclusive", dest="exact", action="store_false")
    step.add_argument("--kill-on-bad-exit", type=int, default=1)
    step.add_argument("command", nargs=argparse.REMAINDER)

    return parser


def build_remote_script(args: argparse.Namespace) -> str:
    env = parse_env_pairs(args.env)
    command = strip_command_prefix(args.command)
    if args.mode == "remote":
        return build_shell_script(command, workdir=args.workdir, env=env)

    step = StepSpec(
        alloc_job_id=args.alloc_job_id,
        nodelist=args.nodelist,
        nodes=args.nodes,
        ntasks=args.ntasks,
        ntasks_per_node=args.ntasks_per_node,
        cpus_per_task=args.cpus_per_task,
        gres=args.gres,
        exact=args.exact,
        exclusive=not args.exact,
        kill_on_bad_exit=args.kill_on_bad_exit,
    )
    step_shell = build_shell_script(command, workdir=args.workdir, env=env)
    return shlex.join(build_srun_command(step, step_shell))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    remote_script = build_remote_script(args)
    relay_cmd = build_falcon_relay_command(
        remote_script,
        tc_host=args.tc_host,
        falcon_host=args.falcon_host,
    )

    if args.dry_run:
        print(render_local_command(relay_cmd))
        return

    rc = run_local_command(relay_cmd, capture_output=args.capture_output)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
