"""Training loops for feed-forward networks."""

import logging, pathlib, timeit
import torch
from datetime import datetime
from tqdm import tqdm

from dlkit.opt.train_utils import (
    checkpoint_path,
    checkpoint_save,
    train_dlog_epoch_initialize,
    train_dlog_epoch_update,
    train_dlog_epoch_finalize,
    train_dlog_batch_initialize,
    train_dlog_batch_update,
    train_dlog_batch_finalize,
)

def train_epochs(
    n_epochs,
    net,
    dataloader,
    optimizer,
    loss_fn,
    validation_fn=None,
    lr_scheduler=None,
    device=None,
    inputs_transform_fn=None,
    targets_transform_fn=None,
    logger=logging.getLogger("dlkit.opt.train_epochs"),
    checkpoint_epochs=None,
    checkpoint_dir="checkpoints",
    epoch_initialize_fn=None,
    epoch_finalize_fn=None,
    grad_clip_max_norm=None,
    reduce_loss_for_logging=True,
    use_amp=False,
    amp_dtype=torch.float16,
    grad_accum_steps=1,
    use_no_sync_for_accum=True,
):
    epoch_dlog = train_dlog_epoch_initialize(n_epochs, ["loss_mean", "loss_std"])

    dist_init = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if dist_init else 0
    world_size = torch.distributed.get_world_size() if dist_init else 1
    is_main = (rank == 0)
    amp_enabled = bool(use_amp and device is not None and getattr(device, "type", None) == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if checkpoint_epochs is not None and not is_main:
        checkpoint_epochs = None

    if checkpoint_epochs is not None:
        assert 1 <= checkpoint_epochs, checkpoint_epochs
        assert checkpoint_dir is not None
        checkpoint_time = datetime.now().strftime("%Y-%m-%d_t%H%M%S")
        checkpoint_dir_ = pathlib.Path(checkpoint_dir) / checkpoint_time
        checkpoint_dir_.mkdir(parents=True, exist_ok=True)

    time_train = timeit.default_timer()
    for epoch_idx in tqdm(range(n_epochs), desc="epochs", disable=(not is_main)):
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch_idx)

        if epoch_initialize_fn:
            epoch_initialize_fn(epoch_idx)

        if checkpoint_epochs is not None and (epoch_idx % checkpoint_epochs == 0):
            path = checkpoint_path(checkpoint_dir_, n_epochs, prefix="net", epoch=epoch_idx)
            logger.debug(f"epoch {epoch_idx:4d}, save checkpoint to '{path}'")
            checkpoint_save(net, path, epoch=epoch_idx, optimizer=optimizer)

        if validation_fn is not None:
            validation_fn(epoch_idx, net)

        batch_dlog = train_batches(
            epoch_idx,
            net,
            dataloader,
            optimizer,
            loss_fn,
            device=device,
            inputs_transform_fn=inputs_transform_fn,
            targets_transform_fn=targets_transform_fn,
            logger=logger,
            grad_clip_max_norm=grad_clip_max_norm,
            reduce_loss_for_logging=reduce_loss_for_logging,
            use_amp=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            grad_accum_steps=grad_accum_steps,
            use_no_sync_for_accum=use_no_sync_for_accum,
        )

        if lr_scheduler is not None:
            lr_current = lr_scheduler.get_last_lr()
            lr_current = f"{lr_current[0]:.6e}" if len(lr_current) == 1 else str(lr_current)
            logger.debug(f"epoch {epoch_idx:4d}, learning_rate {lr_current}")
            try:
                lr_scheduler.step()
            except Exception as exc:
                sched_name = lr_scheduler.__class__.__name__
                raise RuntimeError(
                    f"learning-rate scheduler step failed at epoch {epoch_idx} "
                    f"for {sched_name}; check scheduler stage lengths versus n_epochs"
                ) from exc

        train_dlog_epoch_update(epoch_dlog, epoch_idx, ["loss_mean", "loss_std"], batch_dlog)

        if is_main:
            logger.info(
                f"epoch {epoch_idx:4d}, "
                f"loss mean {batch_dlog['loss_mean']:.6e} std {batch_dlog['loss_std']:.3e}"
            )

        if epoch_finalize_fn:
            epoch_finalize_fn(epoch_idx)

        # Save the fully trained checkpoint immediately after the final epoch
        # completes, so a teardown problem after training does not erase the
        # only useful recovery artifact.
        if checkpoint_epochs is not None and (epoch_idx + 1 == n_epochs):
            path = checkpoint_path(checkpoint_dir_, n_epochs, prefix="net", epoch=n_epochs)
            logger.debug(f"epoch {n_epochs:4d}, save final checkpoint to '{path}'")
            checkpoint_save(net, path, epoch=n_epochs, optimizer=optimizer)

    if checkpoint_epochs is not None:
        path = checkpoint_path(checkpoint_dir_, n_epochs, prefix="net", epoch=n_epochs)
        logger.debug(f"epoch {n_epochs:4d}, save checkpoint to '{path}'")
        checkpoint_save(net, path, epoch=n_epochs, optimizer=optimizer)

    if validation_fn is not None:
        validation_fn(n_epochs, net)

    time_train = timeit.default_timer() - time_train
    train_dlog_epoch_finalize(epoch_dlog, time_train)

    if is_main:
        n_steps = n_epochs * len(dataloader)
        n_samples = n_steps * dataloader.batch_size
        logger.info(f"number of epochs {n_epochs}, optimizer steps {n_steps}, samples processed {n_samples}")
        logger.info(f"training time {time_train:g} sec, time/epoch {time_train / n_epochs:g} sec")
        logger.info(f"time/step {time_train / n_steps:g} sec, samples/sec {n_samples / time_train:g} sec")

    return epoch_dlog


def train_batches(
    epoch_idx,
    net,
    dataloader,
    optimizer,
    loss_fn,
    device=None,
    inputs_transform_fn=None,
    targets_transform_fn=None,
    logger=logging.getLogger("dlkit.opt.train_batches"),
    batch_initialize_fn=None,
    batch_finalize_fn=None,
    max_batches=None,
    grad_clip_max_norm=None,
    reduce_loss_for_logging=True,
    use_amp=False,
    amp_dtype=torch.float16,
    scaler=None,
    grad_accum_steps=1,
    use_no_sync_for_accum=True,
):
    """Runs training loop over batches."""
    if max_batches is None:
        max_batches = len(dataloader)
    batch_dlog = train_dlog_batch_initialize(max_batches, ["loss"], save_list=False)

    # <code id="training_loop_over_batches">
    try:
        grad_accum_steps = int(grad_accum_steps)
    except Exception as exc:
        raise ValueError(f"grad_accum_steps must be int-compatible, got {grad_accum_steps}") from exc
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

    optimizer.zero_grad()
    for batch_idx, data in enumerate(dataloader):
        if max_batches <= batch_idx:
            break

        # initialize batch
        if batch_initialize_fn:
            batch_initialize_fn(batch_idx)

        # set network to training mode
        net.train()

        # get input and target tensors
        inputs, targets = data
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)
        if inputs_transform_fn is not None:
            inputs = inputs_transform_fn(inputs)
        if targets_transform_fn is not None:
            targets = targets_transform_fn(targets)

        # zero the gradients (begin AD)
        optimizer.zero_grad()
        amp_enabled = bool(use_amp and scaler is not None and scaler.is_enabled())

        # forward pass
        step_now = (((batch_idx + 1) % grad_accum_steps) == 0) or ((batch_idx + 1) == max_batches)
        should_use_no_sync = (
            use_no_sync_for_accum
            and (grad_accum_steps > 1)
            and (not step_now)
            and hasattr(net, "no_sync")
        )
        sync_context = net.no_sync if should_use_no_sync else None

        if sync_context is not None:
            context = sync_context()
        else:
            from contextlib import nullcontext
            context = nullcontext()

        with context:
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = net(inputs)
                    loss = loss_fn(outputs, targets)
                    loss_to_backprop = loss / float(grad_accum_steps)
            else:
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)
                loss_to_backprop = loss / float(grad_accum_steps)

            # calculate derivatives (end AD)
            if amp_enabled:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

        # update network parameters only at accumulation boundaries
        if step_now:
            if amp_enabled:
                if grad_clip_max_norm is not None:
                    try:
                        grad_clip_val = float(grad_clip_max_norm)
                    except Exception as exc:
                        raise ValueError(
                            f"grad_clip_max_norm must be a float-compatible value, got {grad_clip_max_norm}"
                        ) from exc
                    if grad_clip_val > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip_max_norm is not None:
                    try:
                        grad_clip_val = float(grad_clip_max_norm)
                    except Exception as exc:
                        raise ValueError(
                            f"grad_clip_max_norm must be a float-compatible value, got {grad_clip_max_norm}"
                        ) from exc
                    if grad_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
                optimizer.step()
            optimizer.zero_grad()

        # log
        loss_det = loss.detach()
        if (
            reduce_loss_for_logging
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(loss_det, op=torch.distributed.ReduceOp.SUM)
            loss_det = loss_det / torch.distributed.get_world_size()

        loss_v = loss_det.item()
        train_dlog_batch_update(batch_dlog, batch_idx, {"loss": loss_v})

        logger.debug(f"batch {batch_idx:4d}, loss {loss_v:.6e}")

        # finalize batch
        if batch_finalize_fn:
            batch_finalize_fn(batch_idx)
    # </code>

    # finalize and return log
    train_dlog_batch_finalize(batch_dlog, ["loss"])
    return batch_dlog
