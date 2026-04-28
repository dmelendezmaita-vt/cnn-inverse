"""Schedulers for learning rates."""

import torch


def create_learning_rate_scheduler(
    optimizer,
    n_epochs,
    learning_rate,
    linear_epochs=None,
    constant_epochs=None,
    init_learning_rate=None,
    final_learning_rate=None,
):
    """
    Creates a schedule for learning rate with these stages:

    1. `init_learning_rate .. learning_rate` for #epochs = `0 .. linear_epochs`
    2. `learning_rate` for #epochs = `linear_epochs .. constant_epochs`
    3. `learning_rate .. final_learning_rate` for #epochs = `constant_epochs .. n_epochs`
    """
    n_epochs = int(n_epochs)
    if n_epochs <= 1:
        # A one-epoch run has no useful decay horizon. Returning None avoids
        # invalid cosine settings (for example T_max=0) in smoke tests.
        return None

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    # Set up defaults.
    if linear_epochs is None:
        linear_epochs = n_epochs // 10
    if constant_epochs is None:
        constant_epochs = n_epochs // 10
    if init_learning_rate is None:
        init_learning_rate = learning_rate / 10.0
    if final_learning_rate is None:
        final_learning_rate = learning_rate / 100.0

    linear_epochs = max(0, int(linear_epochs))
    constant_epochs = max(0, int(constant_epochs))

    # Keep at least one epoch for cosine so T_max stays valid.
    max_pre_epochs = n_epochs - 1
    linear_epochs = min(linear_epochs, max_pre_epochs)
    constant_epochs = min(constant_epochs, max_pre_epochs - linear_epochs)

    schedulers = []
    milestones = []
    completed_epochs = 0

    if linear_epochs > 0:
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=init_learning_rate / learning_rate,
                end_factor=1.0,
                total_iters=linear_epochs,
            )
        )
        completed_epochs += linear_epochs
        milestones.append(completed_epochs)

    if constant_epochs > 0:
        schedulers.append(
            torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=constant_epochs,
            )
        )
        completed_epochs += constant_epochs
        milestones.append(completed_epochs)

    cosine_epochs = max(1, n_epochs - completed_epochs)
    schedulers.append(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=final_learning_rate,
        )
    )

    if len(schedulers) == 1:
        return schedulers[0]

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones[: len(schedulers) - 1],
    )
