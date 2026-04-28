import pathlib
import numpy as np
import torch


def checkpoint_path(checkpoint_dir, n_epochs, prefix, epoch):
    n_digits = int(np.ceil(np.log10(1.01 * n_epochs)))
    fmt = "{}_e{:0" + str(n_digits) + "d}.pt"
    filename = fmt.format(prefix, epoch)
    filepath = pathlib.Path(checkpoint_dir) / filename
    return filepath


def checkpoint_save(model, filepath, epoch, optimizer):
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filepath,
    )



def train_dlog_batch_initialize(n_batches, tags, save_list=False):
    dlog = {"n_batches": n_batches}
    for tag in tags:
        if save_list:
            dlog[tag] = np.empty((n_batches,))
        else:
            dlog[tag] = None
        dlog[f"{tag}_mean_n"] = 0
        dlog[f"{tag}_mean"] = 0.0
        dlog[f"{tag}_sq_mean"] = 0.0
        dlog[f"{tag}_std"] = None
    return dlog


def train_dlog_batch_update(dlog, batch_idx, values):
    for tag in values.keys():
        if tag in values:
            val = values[tag]
        else:
            val = np.nan
        if dlog[tag] is not None:
            dlog[tag][batch_idx] = val
        if not np.isnan(val):
            dlog[f"{tag}_mean_n"] += 1
            dlog[f"{tag}_mean"] += val
            dlog[f"{tag}_sq_mean"] += val * val


def train_dlog_batch_finalize(dlog, tags):
    for tag in tags:
        assert dlog[f"{tag}_std"] is None
        assert not np.isnan(dlog[f"{tag}_mean"])
        assert not np.isnan(dlog[f"{tag}_sq_mean"])
        if 0 < dlog[f"{tag}_mean_n"]:
            dlog[f"{tag}_mean"] *= 1.0 / dlog[f"{tag}_mean_n"]
            dlog[f"{tag}_sq_mean"] *= 1.0 / dlog[f"{tag}_mean_n"]
            dlog[f"{tag}_std"] = np.sqrt(
                dlog[f"{tag}_sq_mean"] - dlog[f"{tag}_mean"] ** 2
            )
        else:
            dlog[f"{tag}_mean"] = 0.0
            dlog[f"{tag}_sq_mean"] = 0.0
            dlog[f"{tag}_std"] = 0.0


def train_dlog_epoch_initialize(n_epochs, tags, save_list=True):
    dlog = {}
    for tag in tags:
        if save_list:
            dlog[tag] = np.empty((n_epochs,))
        else:
            dlog[tag] = None
    dlog["batch_dlog"] = []
    return dlog


def train_dlog_epoch_update(dlog, epoch_idx, tags, batch_dlog):
    for tag in tags:
        if dlog[tag] is not None:
            dlog[tag][epoch_idx] = batch_dlog[tag]
    dlog["batch_dlog"].append(batch_dlog)


def train_dlog_epoch_finalize(dlog, time_train):
    dlog["time_train"] = time_train
