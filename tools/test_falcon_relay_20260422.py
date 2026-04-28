#!/usr/bin/env python3
from __future__ import annotations

from run_falcon_relay_20260422 import (
    StepSpec,
    build_falcon_relay_command,
    build_shell_script,
    build_srun_command,
)


def assert_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"Missing expected text: {needle}\n{text}")


def main() -> None:
    shell_script = build_shell_script(
        ["python", "-c", "print('ok')"],
        workdir="/tmp/falcon relay",
        env={"OMP_NUM_THREADS": "12", "CUDA_VISIBLE_DEVICES": "0"},
    )
    assert_contains(shell_script, "cd '/tmp/falcon relay'")
    assert_contains(shell_script, "export OMP_NUM_THREADS=12")
    assert_contains(shell_script, "export CUDA_VISIBLE_DEVICES=0")
    assert_contains(shell_script, "python -c 'print('\"'\"'ok'\"'\"')'")

    srun_cmd = build_srun_command(
        StepSpec(
            alloc_job_id="368960",
            nodelist="fal117",
            nodes=1,
            ntasks=1,
            ntasks_per_node=1,
            cpus_per_task=12,
            gres="gpu:v100:1",
            exact=True,
        ),
        shell_script,
    )
    srun_rendered = " ".join(srun_cmd)
    assert_contains(srun_rendered, "--jobid=368960")
    assert_contains(srun_rendered, "--exact")
    assert_contains(srun_rendered, "-wfal117")
    assert_contains(srun_rendered, "--cpus-per-task=12")
    assert_contains(srun_rendered, "--gres=gpu:v100:1")

    relay_cmd = build_falcon_relay_command("echo hello")
    relay_rendered = " ".join(relay_cmd)
    assert_contains(relay_rendered, "tinkercliffs1")
    assert_contains(relay_rendered, "falcon1.arc.vt.edu")
    assert_contains(relay_rendered, "bash -lc")
    print("FALCON_RELAY_HELPER_TEST_OK")


if __name__ == "__main__":
    main()
