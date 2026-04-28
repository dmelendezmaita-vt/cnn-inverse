"""Profiling of training steps."""

import logging, os
import torch


def get_table(prof, sort_by, row_limit=10):
    """Prints a table with metrics into stdout.

    Options for sort_by:
    - "cpu_time_total" ... time of functions including subfunctions
    - "self_cpu_time_total" ... time of functions without subfunctions
    - [self_ +] f"{device}_time_total"
    - "self_cpu_memory_usage"
    """
    table = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    table = f"<{sort_by}>\n{table.strip()}\n</{sort_by}>\n"
    return table


def trace_handler(prof, device=None, log_profile_dir="."):
    """Callback function for processing traces.

    alternative handler: torch.profiler.tensorboard_trace_handler(log_profile_dir)
    """
    os.makedirs(log_profile_dir, exist_ok=True)

    # generate tables
    table = f"<profile_result step={prof.step_num}>\n"
    table += get_table(prof, "cpu_time_total")
    table += get_table(prof, "self_cpu_time_total")
    if device:
        table += get_table(prof, f"{device}_time_total")
        table += get_table(prof, f"self_{device}_time_total")
    table += get_table(prof, "self_cpu_memory_usage")
    if device:
        table += get_table(prof, f"self_{device}_memory_usage")
    table += f"</profile_result>\n"
    # write file and print
    table_path = os.path.join(log_profile_dir, f"table_prof_step_{prof.step_num}.txt")
    with open(table_path, "w") as f:
        f.write(table)
    print(table)

    # write trace file
    trace_path = os.path.join(log_profile_dir, f"trace_prof_step_{prof.step_num}.json")
    prof.export_chrome_trace(trace_path)


def profile_train_epochs(
    train_epochs_fn,
    train_epochs_fn_kwargs,
    net,
    dataloader,
    optimizer,
    loss_fn,
    log_profile_dir=".",
):
    """Profiles training over epochs.

    Total number of profiling steps:
        1 wait + 1 warmup + 3*2 active = 8 steps

    Sources:
    - https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    """
    # trace activities
    activities = [torch.profiler.ProfilerActivity.CPU]
    device = None
    if torch.cuda.is_available():
        activities += [torch.profiler.ProfilerActivity.CUDA]
        device = "cuda"
    elif torch.xpu.is_available():
        activities += [torch.profiler.ProfilerActivity.XPU]
        device = "xpu"

    # set schedule
    schedule = torch.profiler.schedule(
        skip_first=0,  # ignore steps
        wait=1,  # inactive during first epoch (at each repeat)
        warmup=1,  # warm-up for 1 epoch
        active=3,  # profile next 3 epochs
        repeat=2,  # repeat this cycle 2 times
    )

    # profile steps
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=lambda p: trace_handler(p, device, log_profile_dir),
        record_shapes=True,  # record shapes of the operator inputs
        profile_memory=True,  # report amount of memory consumed by the model's tensors
        with_stack=True,  # record stack traces
    ) as prof:

        def epoch_finalize_fn(epoch_idx):
            """Signals the end of the step to the profiler."""
            prof.step()

        n_epochs = 10  # a number higher than the #profiling steps
        train_epochs_fn(
            n_epochs,
            net,
            dataloader,
            optimizer,
            loss_fn,
            logger=logging.getLogger("dlkit.opt.profile_train_epochs"),
            epoch_finalize_fn=epoch_finalize_fn,
            **train_epochs_fn_kwargs,
        )

        return prof


def profile_train_batches(
    train_batches_fn,
    train_batches_fn_kwargs,
    net,
    dataloader,
    optimizer,
    loss_fn,
    log_profile_dir=".",
    logger=logging.getLogger("dlkit.opt.profile_train_batches"),
):
    """Profiles training over batches.

    Total number of profiling steps:
        (1 wait + 1 warmup + 3 active) * 2 repeats = 10 steps

    Sources:
    - https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    """
    # trace activities
    activities = [torch.profiler.ProfilerActivity.CPU]
    device = None
    if torch.cuda.is_available():
        activities += [torch.profiler.ProfilerActivity.CUDA]
        device = "cuda"
    elif torch.xpu.is_available():
        activities += [torch.profiler.ProfilerActivity.XPU]
        device = "xpu"

    # set schedule
    schedule = torch.profiler.schedule(
        skip_first=0,  # ignore steps
        wait=1,  # inactive during first epoch (at each repeat)
        warmup=1,  # warm-up for 1 epoch
        active=3,  # profile next 3 epochs
        repeat=2,  # repeat this cycle 2 times
    )
    max_batches = 10  # a number higher than the #profiling steps

    # profile steps
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=lambda p: trace_handler(p, device, log_profile_dir),
        record_shapes=True,  # record shapes of the operator inputs
        profile_memory=True,  # report amount of memory consumed by the model's tensors
        with_stack=True,  # record stack traces
    ) as prof:

        def batch_finalize_fn(batch_idx):
            """Signals the end of the step to the profiler."""
            prof.step()

        epoch_idx = 0
        batch_dlog = train_batches_fn(
            epoch_idx,
            net,
            dataloader,
            optimizer,
            loss_fn,
            logger=logger,
            batch_finalize_fn=batch_finalize_fn,
            max_batches=max_batches,
            **train_batches_fn_kwargs,
        )
        n_batches = len(batch_dlog["loss"])

        if n_batches < max_batches:
            logger.warning(
                f"Expected {max_batches} batches for profiling, got {n_batches} batches"
            )

        return prof
