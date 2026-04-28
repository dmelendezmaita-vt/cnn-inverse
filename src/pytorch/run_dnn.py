"""
Run training and evaluation of DNN-based inverse map.
"""

import argparse, os, pprint, random, sys, time, timeit, signal, tempfile
from datetime import timedelta
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
    PostLocalSGDState,
    post_localSGD_hook,
)
from torch.distributed.algorithms.model_averaging.averagers import (
    PeriodicModelAverager,
)
from torch.distributed.optim import PostLocalSGDOptimizer
from tqdm import tqdm

from dlkit.log.log_util import (
    logging_set_up,
    logging_get_logger
)
from dlkit.nets.util import get_parameters
from dlkit.opt.train import train_epochs

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

from utils import (
    Mode,
    load_parameters,
    save_parameters,
    update_parameters_from_args,
    plot_loss,
    plot_data_vs_predict,
    plot_data_vs_predict_error
)
from data import (
    dictarray_is_not_none,
    dictarray_uses_indexed_arrays,
    load_data,
    make_features_preprocess_transform,
    make_targets_preprocess_transform,
    preprocess_features,
    preprocess_targets,
    postprocess_targets,
    create_dataloader
)
from nets import (
    create_network,
    create_ae
)
from opt_utils import (
    create_optimizer,
    create_lr_scheduler
)

###############################################################################

def _dist_from_env():
    """
    Returns (world_size, rank, local_rank).

    Supports torchrun-style env vars:
      WORLD_SIZE, RANK, LOCAL_RANK

    Supports Slurm-style env vars:
      SLURM_NTASKS, SLURM_PROCID, SLURM_LOCALID
    """
    # torchrun path
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return world_size, rank, local_rank

    # Slurm path
    if "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        # populate torchrun-compatible names so init_method="env://" works
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        return world_size, rank, local_rank

    # non-distributed
    return 1, 0, 0


def _unwrap_ddp(model):
    return model.module if isinstance(model, DDP) else model

def _install_slurm_handlers(net, optimizer, lr_scheduler, checkpoint_dir, is_main, epoch_ref=None):
    """
    On SIGUSR1 (sent by Slurm pre-timeout) or SIGTERM, save a checkpoint and exit.
    Only rank 0 writes the checkpoint to avoid corruption under DDP.
    """
    if checkpoint_dir is None:
        return

    def _handler(signum, frame):
        try:
            if is_main:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(checkpoint_dir, f"net_interrupt_{ts}.pt")

                base = _unwrap_ddp(net)

                epoch_val = -1
                if isinstance(epoch_ref, dict) and (epoch_ref.get("epoch") is not None):
                    epoch_val = int(epoch_ref["epoch"])

                payload = {
                    "epoch": epoch_val,
                    "model_state_dict": base.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                if lr_scheduler is not None:
                    payload["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

                torch.save(payload, path)
                print(f"[signal] saved interrupt checkpoint: {path}", flush=True)
        finally:
            sys.exit(0)

    signal.signal(signal.SIGUSR1, _handler)
    signal.signal(signal.SIGTERM, _handler)


class WeightedMSELoss(torch.nn.Module):
    """
    Per-target weighted MSE.
    Expected inputs:
      pred:   [batch, n_targets]
      target: [batch, n_targets]
    """
    def __init__(self, weights):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32).view(1, -1)
        self.register_buffer("weights", w)

    def forward(self, pred, target):
        if pred.ndim < 2 or target.ndim < 2:
            raise ValueError(
                f"WeightedMSELoss expects 2D tensors [batch, n_targets], got pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        if pred.shape[1] != self.weights.shape[1]:
            raise ValueError(
                f"WeightedMSELoss target dimension mismatch: pred.shape[1]={pred.shape[1]} "
                f"!= len(weights)={self.weights.shape[1]}"
            )
        err2 = (pred - target) ** 2
        return (err2 * self.weights).mean()


class WeightedSmoothL1Loss(torch.nn.Module):
    """
    Elementwise SmoothL1/Huber loss with optional per-target weights.
    """

    def __init__(self, weights: torch.Tensor, beta: float = 1.0):
        super().__init__()
        if weights.ndim == 1:
            weights = weights.view(1, -1)
        self.register_buffer("weights", weights)
        self.beta = float(beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 2 or target.ndim != 2:
            raise ValueError(
                f"WeightedSmoothL1Loss expects 2D tensors [batch, n_targets], got pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        if pred.shape[1] != self.weights.shape[1]:
            raise ValueError(
                f"WeightedSmoothL1Loss target dimension mismatch: pred.shape[1]={pred.shape[1]} "
                f"!= len(weights)={self.weights.shape[1]}"
            )
        loss = torch.nn.functional.smooth_l1_loss(
            pred,
            target,
            beta=self.beta,
            reduction="none",
        )
        return (loss * self.weights).mean()


class UncertaintyWeightedMultiTaskLoss(torch.nn.Module):
    """
    Multi-task uncertainty weighting with learnable per-target log variances.

    The log-variance parameters live on the network (`net.log_task_vars`) so
    they are optimized and synchronized together with the model parameters.
    """

    def __init__(self, log_vars: torch.nn.Parameter, *, base_loss: str = "huber", beta: float = 1.0, static_weights=None):
        super().__init__()
        self.log_vars = log_vars
        self.base_loss = str(base_loss).strip().casefold()
        self.beta = float(beta)
        if static_weights is None:
            self.static_weights = None
        else:
            w = torch.as_tensor(static_weights, dtype=torch.float32)
            if w.ndim == 1:
                w = w.view(1, -1)
            self.register_buffer("static_weights", w)

    def _elementwise_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.base_loss == "mse":
            return (pred - target) ** 2
        if self.base_loss in ("smoothl1", "huber"):
            return torch.nn.functional.smooth_l1_loss(
                pred,
                target,
                beta=self.beta,
                reduction="none",
            )
        raise ValueError(f"Unknown base_loss for uncertainty weighting: {self.base_loss}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 2 or target.ndim != 2:
            raise ValueError(
                f"UncertaintyWeightedMultiTaskLoss expects 2D tensors [batch, n_targets], got pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        if pred.shape[1] != self.log_vars.shape[0]:
            raise ValueError(
                f"UncertaintyWeightedMultiTaskLoss target dimension mismatch: pred.shape[1]={pred.shape[1]} "
                f"!= len(log_vars)={self.log_vars.shape[0]}"
            )
        base = self._elementwise_loss(pred, target)
        if self.static_weights is not None:
            if base.shape[1] != self.static_weights.shape[1]:
                raise ValueError(
                    f"UncertaintyWeightedMultiTaskLoss weight dimension mismatch: base.shape[1]={base.shape[1]} "
                    f"!= len(weights)={self.static_weights.shape[1]}"
                )
            base = base * self.static_weights
        per_target = base.mean(dim=0)
        weighted = 0.5 * torch.exp(-self.log_vars) * per_target + 0.5 * self.log_vars
        return weighted.mean()


def _build_target_loss_weights(params, targets_scale, logger):
    """
    Build optional per-target loss weights.

    Controls (under data):
      - target_loss_weight_mode: uniform | manual | inverse_std | inverse_var
      - target_loss_weights: [w1, ..., wn] (used when mode=manual, or implied if provided)
      - target_loss_weights_normalize: bool (default True; mean weight -> 1)
    """
    n_targets = int(params.get("data", {}).get("num_targets", 0) or 0)
    if n_targets <= 0:
        return None

    data_cfg = params.get("data", {})
    mode = str(data_cfg.get("target_loss_weight_mode", "uniform")).strip().casefold()
    manual = data_cfg.get("target_loss_weights")
    if manual is not None and mode == "uniform":
        mode = "manual"

    normalize = bool(data_cfg.get("target_loss_weights_normalize", True))

    if mode == "uniform":
        logger.info("Loss weighting - mode=uniform (plain MSE)")
        return None

    if mode == "manual":
        if manual is None:
            raise ValueError("data.target_loss_weight_mode='manual' requires data.target_loss_weights")
        w = np.asarray(manual, dtype=np.float32).reshape(-1)
        if w.shape[0] != n_targets:
            raise ValueError(
                f"data.target_loss_weights length mismatch: got {w.shape[0]}, expected {n_targets}"
            )
    elif mode in ("inverse_std", "inverse_var"):
        if not isinstance(targets_scale, dict) or ("mult" not in targets_scale):
            raise ValueError(
                f"data.target_loss_weight_mode='{mode}' requires targets_scale['mult'] from preprocess_targets"
            )
        std = np.asarray(targets_scale["mult"], dtype=np.float32).reshape(-1)
        if std.shape[0] != n_targets:
            raise ValueError(
                f"targets_scale['mult'] length mismatch: got {std.shape[0]}, expected {n_targets}"
            )
        std = np.maximum(std, 1.0e-12)
        if mode == "inverse_std":
            w = 1.0 / std
        else:
            w = 1.0 / (std ** 2)
    else:
        raise ValueError(
            f"Unknown data.target_loss_weight_mode='{mode}'. "
            "Supported: uniform, manual, inverse_std, inverse_var"
        )

    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError(f"Invalid target loss weights computed for mode='{mode}': {w}")

    if normalize:
        w = w * (float(n_targets) / float(np.sum(w)))

    logger.info(
        "Loss weighting - mode=%s normalize=%s weights=%s",
        mode,
        normalize,
        w.tolist(),
    )
    return torch.as_tensor(w, dtype=torch.float32)


def _maybe_wrap_post_local_sgd(net, optimizer, params, dist_on, logger):
    """
    Optionally enable PyTorch post-local SGD for distributed DDP training.

    This registers the post-local communication hook on the DDP-wrapped model
    and wraps the base optimizer in PostLocalSGDOptimizer so periodic global
    model averaging occurs after optimizer steps.
    """
    cfg = params.get("runconfig", {})
    enabled = bool(cfg.get("post_local_sgd_enabled", False))
    if not enabled:
        return optimizer

    if not dist_on:
        logger.warning(
            "runconfig.post_local_sgd_enabled=True ignored because distributed training is off."
        )
        return optimizer

    if not isinstance(net, DDP):
        raise RuntimeError("post-local SGD requires the training network to be wrapped in DDP.")

    warmup_steps = int(cfg.get("post_local_sgd_warmup_steps", 100))
    period = int(cfg.get("post_local_sgd_period", 4))
    start_local_iter = int(cfg.get("post_local_sgd_start_local_sgd_iter", warmup_steps))
    post_local_gradient_allreduce = bool(cfg.get("post_local_gradient_allreduce", True))

    if warmup_steps < 0:
        raise ValueError(
            f"runconfig.post_local_sgd_warmup_steps must be >= 0, got {warmup_steps}"
        )
    if period < 1:
        raise ValueError(f"runconfig.post_local_sgd_period must be >= 1, got {period}")
    if start_local_iter < 0:
        raise ValueError(
            "runconfig.post_local_sgd_start_local_sgd_iter must be >= 0, "
            f"got {start_local_iter}"
        )
    if start_local_iter != warmup_steps:
        raise ValueError(
            "runconfig.post_local_sgd_start_local_sgd_iter must match "
            "runconfig.post_local_sgd_warmup_steps for PeriodicModelAverager consistency."
        )

    state = PostLocalSGDState(
        process_group=None,
        subgroup=None,
        start_localSGD_iter=start_local_iter,
        post_local_gradient_allreduce=post_local_gradient_allreduce,
    )
    net.register_comm_hook(state, post_localSGD_hook)

    averager = PeriodicModelAverager(
        period=period,
        warmup_steps=warmup_steps,
        process_group=None,
    )
    logger.info(
        "Enabled post-local SGD - warmup_steps=%s period=%s post_local_gradient_allreduce=%s",
        warmup_steps,
        period,
        post_local_gradient_allreduce,
    )
    return PostLocalSGDOptimizer(optimizer, averager)


def _configure_rank_tmpdir(local_rank, logger):
    """
    Use a rank-local temp directory so multiprocessing teardown does not race
    across distributed ranks sharing the same launcher step.
    """
    base_tmp = os.environ.get("TMPDIR")
    if not base_tmp:
        return
    rank_tmp = os.path.join(base_tmp, f"localrank_{local_rank}")
    os.makedirs(rank_tmp, exist_ok=True)
    os.environ["TMPDIR"] = rank_tmp
    os.environ["TMP"] = rank_tmp
    os.environ["TEMP"] = rank_tmp
    tempfile.tempdir = rank_tmp
    logger.info("Runtime temp dir - rank-local TMPDIR=%s", rank_tmp)


def _compose_batch_transforms(*funcs):
    funcs = [fn for fn in funcs if fn is not None]
    if not funcs:
        return None

    def _composed(batch):
        out = batch
        for fn in funcs:
            out = fn(out)
        return out

    return _composed



def run(args, params):
    # set environment
    self_dir = os.path.dirname(os.path.abspath(__file__))
    enable_debug = params['runconfig'].get('debug')
    bootstrap_logger = logging_get_logger('run')

    # ----------------------------
    # distributed init (if any)
    # ----------------------------
    world_size, rank, local_rank = _dist_from_env()
    dist_on = (world_size > 1)
    is_main = (rank == 0)

    if dist_on and (not dist.is_available()):
        raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1 was detected")

    _configure_rank_tmpdir(local_rank, bootstrap_logger)

    # pick device per process before distributed init (important for NCCL rank/device mapping)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if dist_on and (not dist.is_initialized()):
        backend = "nccl" if device.type == "cuda" else "gloo"
        timeout_seconds = int(os.environ.get("TORCH_DIST_TIMEOUT_SECONDS", "1800"))
        if timeout_seconds < 60:
            timeout_seconds = 60
        init_kwargs = dict(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=timeout_seconds),
        )
        if backend == "nccl" and device.type == "cuda":
            # Preferred path in newer PyTorch versions; fallback below keeps compatibility.
            try:
                dist.init_process_group(device_id=device, **init_kwargs)
            except TypeError:
                dist.init_process_group(**init_kwargs)
        else:
            dist.init_process_group(**init_kwargs)

    # Fail fast if collectives are unhealthy before expensive training begins.
    if dist_on:
        try:
            probe = torch.tensor([float(rank + 1)], device=device)
            dist.all_reduce(probe, op=dist.ReduceOp.SUM)
            if device.type == "cuda":
                try:
                    dist.barrier(device_ids=[local_rank])
                except TypeError:
                    dist.barrier()
            else:
                dist.barrier()
        except Exception as exc:
            raise RuntimeError("Distributed preflight failed before training; aborting run") from exc

    # set mode
    mode_name = params['runconfig']['mode']
    mode = None
    for name in mode_name.split("_"):
        m = Mode[name.upper()]
        mode = m if mode is None else mode | m

    # set key for data
    if Mode.TRAIN in mode:
        mode_to_data_key = 'train'
    elif mode.any(Mode.PREDICT | Mode.EVAL):
        mode_to_data_key = 'test'
    else:
        raise NotImplementedError()

    # Optional global batch semantics for distributed training:
    # keep effective global train batch approximately constant as world_size changes.
    gbs_requested = params['data'].get('global_train_batch_size')
    gbs_effective = None
    if (Mode.TRAIN in mode) and (gbs_requested is not None):
        gbs_requested = int(gbs_requested)
        if gbs_requested <= 0:
            raise ValueError(f"data.global_train_batch_size must be > 0, got {gbs_requested}")
        per_rank_bs = max(1, gbs_requested // max(1, world_size))
        gbs_effective = per_rank_bs * max(1, world_size)
        params['data']['train_batch_size'] = per_rank_bs
        params['data']['train_batch_size_per_rank'] = per_rank_bs
        params['data']['effective_global_train_batch_size'] = gbs_effective

    # set up logging (separate per-rank directory to avoid file collisions)
    log_root = os.path.join(self_dir, params['runconfig']['save_dir'], "run_dnn")
    log_dir = os.path.join(log_root, f"rank{rank}")
    os.makedirs(log_dir, exist_ok=True)
    logging_set_up(log_dir)
    logger = logging_get_logger(f'run_dnn.rank{rank}')

    # log environment
    cpu_logical_cores = os.cpu_count()
    logger.info(f"Environment - Directory:         {self_dir}")
    logger.info(f"Environment - PyTorch version:   {torch.__version__}")
    logger.info(f"Environment - Seed (base):       {params['data'].get('random_seed')}")
    logger.info(f"Environment - Mode:              {mode} (--mode {mode_name})")
    logger.info(f"Environment - Data key:          {mode_to_data_key}")
    logger.info(f"Environment - CPU logical cores: {cpu_logical_cores}")
    logger.info(f"Environment - Torch device:      {device}")
    logger.info(f"Environment - dist_on:           {dist_on} world_size={world_size} rank={rank} local_rank={local_rank}")
    if (Mode.TRAIN in mode) and (gbs_requested is not None):
        logger.info(
            "Batching - global_train_batch_size requested=%s effective=%s per_rank_train_batch_size=%s",
            gbs_requested,
            gbs_effective,
            params['data']['train_batch_size'],
        )
        if gbs_effective != gbs_requested:
            logger.warning(
                "Batching - requested global batch %s is not divisible by world_size=%s; "
                "using effective global batch %s",
                gbs_requested,
                world_size,
                gbs_effective,
            )

    # print parameters
    if enable_debug and is_main:
        print('<parameters>', flush=True)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
        print('</parameters>', flush=True)

    # fix random seed for reproducibility (offset by rank to avoid identical RNG streams)
    if 'random_seed' in params['data'] and params['data']['random_seed'] is not None:
        base_seed = int(params['data']['random_seed'])
        seed = base_seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        params['data']['random_seed'] = None

    # initialize timers
    time_train = 0.0
    time_eval  = 0.0
    save_diagnostic_plots = bool(
        params.get("runconfig", {}).get("save_diagnostic_plots", True)
    )

    try:
        #
        # Data
        #

        # Persist exact train/validate indices chosen by data.py (after filtering/weighting)
        params["data"]["split_indices_path"] = os.path.join(
            params["runconfig"]["save_dir"], "split_indices.npz"
        )

        # progress markers around the expensive I/O boundary (rank 0 only)
        if is_main:
            print("<progress:begin_data_load>", flush=True)

        features, targets, features_noise, targets_noise = load_data(params, logging_get_logger('load_data'))

        if is_main:
            print("<progress:end_data_load>", flush=True)

        # preprocess data
        t_preprocess_total = timeit.default_timer()
        t_pre = timeit.default_timer()
        features_scale = preprocess_features(features, params, logging_get_logger('preprocess_features'))
        time_preprocess_features = timeit.default_timer() - t_pre

        t_pre = timeit.default_timer()
        targets_scale = preprocess_targets(targets, params, logging_get_logger('preprocess_targets'))
        time_preprocess_targets = timeit.default_timer() - t_pre

        t_pre = timeit.default_timer()
        features_noise_scale = preprocess_features(features_noise, params,
                                                   logging_get_logger('preprocess_features_noise'),
                                                   scale=features_scale, array_name='features_noise')
        time_preprocess_features_noise = timeit.default_timer() - t_pre

        t_pre = timeit.default_timer()
        targets_noise_scale = preprocess_targets(targets_noise, params,
                                                 logging_get_logger('preprocess_targets_noise'),
                                                 array_name='targets_noise')
        time_preprocess_targets_noise = timeit.default_timer() - t_pre
        time_preprocess_total = timeit.default_timer() - t_preprocess_total

        logger.info(f"Runtime - preprocess_features [sec]: {time_preprocess_features}")
        logger.info(f"Runtime - preprocess_targets [sec]:  {time_preprocess_targets}")
        logger.info(f"Runtime - preprocess_features_noise [sec]: {time_preprocess_features_noise}")
        logger.info(f"Runtime - preprocess_targets_noise [sec]:  {time_preprocess_targets_noise}")
        logger.info(f"Runtime - preprocess_total [sec]:  {time_preprocess_total}")

        # set transform functions
        lazy_feature_preprocess = dictarray_uses_indexed_arrays(features)
        lazy_target_preprocess = dictarray_uses_indexed_arrays(targets)

        features_transform_fn = None  # used in dataloader
        targets_transform_fn = None  # used in dataloader
        train_input_transform_fn = None  # used in training loops

        if lazy_feature_preprocess:
            features_transform_fn = make_features_preprocess_transform(features_scale, params)

        if lazy_target_preprocess:
            merged_target_scale = targets_scale
            if targets_scale is not None and targets_noise_scale is not None:
                merged_target_scale = {
                    "shift": np.concatenate((targets_scale["shift"], targets_noise_scale["shift"]), axis=1),
                    "mult": np.concatenate((targets_scale["mult"], targets_noise_scale["mult"]), axis=1),
                    "transform": {
                        "names": list(targets_scale.get("transform", {}).get("names", []))
                        + list(targets_noise_scale.get("transform", {}).get("names", [])),
                        "eps": float(
                            max(
                                targets_scale.get("transform", {}).get("eps", 1.0e-12),
                                targets_noise_scale.get("transform", {}).get("eps", 1.0e-12),
                            )
                        ),
                    },
                }
            targets_transform_fn = make_targets_preprocess_transform(merged_target_scale)

        if params['data'].get('features_fft'):
            def _fft_features_transform(features):
                features_half = features[..., ::2]
                size = features_half.size()
                features_fft = torch.fft.rfft(features, dim=-1, norm="ortho")
                features_fft = features_fft[..., :size[-1]]
                features_transformed = torch.concatenate(
                    (features_half, features_fft.real, features_fft.imag), axis=1
                )
                return features_transformed
            features_transform_fn = _compose_batch_transforms(features_transform_fn, _fft_features_transform)

        if params['data'].get('autoencoder_load_dir'):
            import glob
            import yaml

            requested_param_file = os.path.join(params['data']['autoencoder_load_dir'], 'params.yaml')

            checkpoint_folders = glob.glob(os.path.join(params['data']['autoencoder_load_dir'], "checkpoints", "*"))
            latest_folder = max(checkpoint_folders, key=os.path.getmtime)
            checkpoint_files = glob.glob(os.path.join(latest_folder, "*.pt"))
            requested_checkpoint = max(checkpoint_files, key=os.path.getmtime)

            logger.info(f"Load autoencoder: use parameter file: {requested_param_file}")
            logger.info(f"Load autoencoder: use checkpoint file: {requested_checkpoint}")

            with open(requested_param_file, 'r') as file:
                ae_params = yaml.safe_load(file)
            ae_params['data']['num_features'] = params['data']['num_features']

            autoencoder = create_ae(ae_params, logging_get_logger('create_autoencoder'))
            checkpoint  = torch.load(requested_checkpoint, map_location=device)
            autoencoder.load_state_dict(checkpoint['model_state_dict'])
            autoencoder.to(device)
            autoencoder.eval()

            if is_main:
                print('<autoencoder>', flush=True)
                print(autoencoder, flush=True)
                print('</autoencoder>', flush=True)

            train_input_transform_fn = autoencoder.e_net

        # create dataloader (distributed sampler for training only)
        dataloader = create_dataloader(
            params,
            logging_get_logger('create_dataloader'),
            mode,
            features=features[mode_to_data_key],
            targets=targets[mode_to_data_key],
            features_noise=features_noise[mode_to_data_key],
            targets_noise=targets_noise[mode_to_data_key],
            features_transform_fn=features_transform_fn,
            targets_transform_fn=targets_transform_fn,
            item_return_order='yx',
            distributed=(dist_on and (Mode.TRAIN in mode)),
            rank=rank,
            world_size=world_size,
        )

        #
        # Network
        #

        resume_ckpt_path = None
        start_epoch = 0
        if params['runconfig']['load_dir']:
            resume_ckpt_path = os.path.join(self_dir, params['runconfig']['load_dir'])


        net = create_network(params, logging_get_logger('create_network'))

        # load network weights on rank 0 only (DDP construction will sync parameters)
        if resume_ckpt_path and (is_main or (not dist_on)):
            ckpt0 = torch.load(resume_ckpt_path, map_location="cpu")

            # accept either raw state_dict or dlkit checkpoint dict
            model_sd = ckpt0.get("model_state_dict", ckpt0) if isinstance(ckpt0, dict) else ckpt0

            # strip "module." if it exists
            if isinstance(model_sd, dict) and any(k.startswith("module.") for k in model_sd.keys()):
                model_sd = {k.replace("module.", "", 1): v for k, v in model_sd.items()}

            net.load_state_dict(model_sd)


        net.to(device)

        # wrap in DDP if needed
        if dist_on:
            ddp_cfg = params.get("runconfig", {})
            ddp_static_graph = bool(ddp_cfg.get("ddp_static_graph", False))
            ddp_gradient_as_bucket_view = bool(
                ddp_cfg.get("ddp_gradient_as_bucket_view", False)
            )
            ddp_bucket_cap_mb = ddp_cfg.get("ddp_bucket_cap_mb", None)
            ddp_kwargs = dict(
                device_ids=[local_rank] if device.type == "cuda" else None,
                broadcast_buffers=False,
                static_graph=ddp_static_graph,
                gradient_as_bucket_view=ddp_gradient_as_bucket_view,
            )
            if ddp_bucket_cap_mb is not None:
                try:
                    ddp_kwargs["bucket_cap_mb"] = int(ddp_bucket_cap_mb)
                except Exception as exc:
                    raise ValueError(
                        f"runconfig.ddp_bucket_cap_mb must be int-compatible, got {ddp_bucket_cap_mb}"
                    ) from exc
            net = DDP(
                net,
                **ddp_kwargs,
            )

        # log network and parameters (rank 0 only)
        if is_main:
            base_net = _unwrap_ddp(net)
            n_trainable_params, n_nontrainable_params, net_params_table = get_parameters(base_net)
            net_out_path = os.path.join(self_dir, params['runconfig']['save_dir'], 'net.txt')
            net_out = f"<network>\n{base_net}\n</network>\n"
            net_out += f"<parameters>\n{net_params_table}\n</parameters>\n"
            with open(net_out_path, "w") as f:
                f.write(net_out)
            if enable_debug:
                print(net_out, flush=True)

        #
        # Training
        #

        train_dlog = None
        if Mode.TRAIN in mode:
            if Mode.PROFILE in mode and dist_on:
                raise RuntimeError("Profiling mode is not supported with distributed training enabled")

            optimizer = create_optimizer(net, params['optimizer'])
            optimizer = _maybe_wrap_post_local_sgd(net, optimizer, params, dist_on, logger)
            lr_scheduler = create_lr_scheduler(optimizer, params['optimizer'], params['training']['epochs'])
            target_weights = _build_target_loss_weights(params, targets_scale, logger)
            loss_name = str(params.get("training", {}).get("loss_type", "mse")).strip().casefold()
            reduce_loss_for_logging = bool(
                params.get("training", {}).get("reduce_loss_for_logging", True)
            )
            use_amp = bool(params.get("training", {}).get("use_amp", False))
            grad_accum_steps = int(params.get("training", {}).get("grad_accum_steps", 1))
            use_no_sync_for_accum = bool(
                params.get("training", {}).get("use_no_sync_for_accum", True)
            )
            amp_dtype_name = str(
                params.get("training", {}).get("amp_dtype", "float16")
            ).strip().casefold()
            amp_dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            if amp_dtype_name not in amp_dtype_map:
                raise ValueError(
                    f"Unknown training.amp_dtype='{amp_dtype_name}'. "
                    "Supported: float16/fp16/half, bfloat16/bf16"
                )
            amp_dtype = amp_dtype_map[amp_dtype_name]
            logger.info(
                "Training options - loss_type=%s reduce_loss_for_logging=%s use_amp=%s amp_dtype=%s grad_accum_steps=%s use_no_sync_for_accum=%s",
                loss_name,
                reduce_loss_for_logging,
                use_amp,
                amp_dtype_name,
                grad_accum_steps,
                use_no_sync_for_accum,
            )
            if loss_name in ("mse", "weighted_mse"):
                if target_weights is None:
                    loss_fn = torch.nn.MSELoss()
                else:
                    loss_fn = WeightedMSELoss(target_weights.to(device))
            elif loss_name in ("uncertainty_mse", "uncertainty_weighted_mse"):
                base_net = _unwrap_ddp(net)
                if not hasattr(base_net, "log_task_vars"):
                    raise ValueError(
                        "training.loss_type='uncertainty_mse' requires net.learn_task_uncertainty=True"
                    )
                loss_fn = UncertaintyWeightedMultiTaskLoss(
                    base_net.log_task_vars,
                    base_loss="mse",
                    static_weights=target_weights.to(device) if target_weights is not None else None,
                )
            elif loss_name in ("smoothl1", "huber"):
                beta = float(params.get("training", {}).get("huber_beta", 1.0))
                if target_weights is None:
                    loss_fn = torch.nn.SmoothL1Loss(beta=beta)
                else:
                    loss_fn = WeightedSmoothL1Loss(target_weights.to(device), beta=beta)
            elif loss_name in ("uncertainty_huber", "uncertainty_smoothl1"):
                base_net = _unwrap_ddp(net)
                if not hasattr(base_net, "log_task_vars"):
                    raise ValueError(
                        "training.loss_type='uncertainty_huber' requires net.learn_task_uncertainty=True"
                    )
                beta = float(params.get("training", {}).get("huber_beta", 1.0))
                loss_fn = UncertaintyWeightedMultiTaskLoss(
                    base_net.log_task_vars,
                    base_loss="huber",
                    beta=beta,
                    static_weights=target_weights.to(device) if target_weights is not None else None,
                )
            else:
                raise ValueError(
                    f"Unknown training.loss_type='{loss_name}'. "
                    "Supported: mse, weighted_mse, smoothl1, huber, uncertainty_mse, uncertainty_huber"
                )
            loss_fn = loss_fn.to(device)

            # Resume optimizer/scheduler and epoch index on ALL ranks (each rank has its own optimizer state)
            if resume_ckpt_path:
                ckpt = torch.load(resume_ckpt_path, map_location="cpu")
                if isinstance(ckpt, dict):
                    if "optimizer_state_dict" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if (lr_scheduler is not None) and ("lr_scheduler_state_dict" in ckpt):
                        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
                    if "epoch" in ckpt:
                        start_epoch = int(ckpt["epoch"]) + 1

            checkpoint_dir    = os.path.join(self_dir, params['runconfig']['save_dir'], 'checkpoints')
            checkpoint_epochs = params['runconfig']['save_checkpoints_epochs']
            if checkpoint_epochs is not None:
                try:
                    checkpoint_epochs = int(checkpoint_epochs)
                except Exception as exc:
                    raise ValueError(
                        f"runconfig.save_checkpoints_epochs must be int-compatible or null, got {checkpoint_epochs}"
                    ) from exc
                if checkpoint_epochs <= 0:
                    checkpoint_epochs = None
            if not is_main:
                checkpoint_epochs = None
                checkpoint_dir = None

            train_sampler = getattr(dataloader, "sampler", None)
            epoch_ref = {"epoch": None}
            _install_slurm_handlers(net, optimizer, lr_scheduler, checkpoint_dir, is_main, epoch_ref)


            def _epoch_init(epoch_idx):
                global_epoch = start_epoch + epoch_idx
                epoch_ref["epoch"] = global_epoch

                # required for DistributedSampler shuffling across epochs
                if dist_on and hasattr(train_sampler, "set_epoch"):
                    train_sampler.set_epoch(global_epoch)

            if is_main:
                print(f"<train>", flush=True)

            remaining_epochs = max(0, params['training']['epochs'] - start_epoch)

            if remaining_epochs > 0:
                train_dlog = train_epochs(
                    remaining_epochs,
                    net,
                    dataloader,
                    optimizer,
                    loss_fn,
                    lr_scheduler        = lr_scheduler,
                    device              = device,
                    inputs_transform_fn = train_input_transform_fn,
                    checkpoint_epochs   = checkpoint_epochs,
                    checkpoint_dir      = checkpoint_dir,
                    epoch_initialize_fn = _epoch_init,
                    grad_clip_max_norm  = params.get("training", {}).get("grad_clip_max_norm", None),
                    reduce_loss_for_logging = reduce_loss_for_logging,
                    use_amp             = use_amp,
                    amp_dtype           = amp_dtype,
                    grad_accum_steps    = grad_accum_steps,
                    use_no_sync_for_accum = use_no_sync_for_accum,
                )
                time_train = train_dlog.get('time_train')
            else:
                train_dlog = {}
                time_train = 0.0


        # ensure all ranks complete training before rank 0 evaluates, then
        # tear down distributed state before entering single-rank evaluation.
        if dist_on:
            skip_final_barrier = bool(
                params.get("runconfig", {}).get("skip_final_distributed_barrier", False)
            )
            if skip_final_barrier:
                logger.warning(
                    "Distributed finalize - skipping post-train barrier by configuration"
                )
            else:
                logger.info("Distributed finalize - entering post-train barrier")
                if device.type == "cuda":
                    try:
                        dist.barrier(device_ids=[local_rank])
                    except TypeError:
                        dist.barrier()
                else:
                    dist.barrier()
                logger.info("Distributed finalize - post-train barrier passed")

            if dist.is_initialized():
                logger.info("Distributed finalize - destroying process group")
                dist.destroy_process_group()

            if not is_main:
                return

            logger.info("Distributed finalize - rank0 entering single-rank evaluation")
            net = _unwrap_ddp(net)

        #
        # Prediction
        #

        if not mode.any(Mode.PREDICT | Mode.EVAL):
            return

        print('<predict>', flush=True)

        eval_dataloader = dict()
        for key in features.keys():
            eval_dataloader[key] = create_dataloader(
                params,
                logging_get_logger('create_dataloader'),
                Mode.EVAL,
                features=features[key],
                targets=targets[key],
                features_noise=features_noise[key],
                targets_noise=targets_noise[key],
                features_transform_fn=features_transform_fn,
                targets_transform_fn=targets_transform_fn,
                item_return_order='yx',
                distributed=False,
                rank=0,
                world_size=1,
            )

        time_eval = timeit.default_timer()
        eval_targets_pred, eval_targets_data = predict(net, eval_dataloader, params, device,
                                                       input_transform_fn=train_input_transform_fn)
        time_eval = timeit.default_timer() - time_eval

        # postprocess evaluation data
        if dictarray_is_not_none(targets) and dictarray_is_not_none(targets_noise):
            eval_targets_scale = {}
            for key in targets_scale.keys():
                eval_targets_scale[key] = np.concatenate((targets_scale[key], targets_noise_scale[key]), axis=1)
            postprocess_targets(eval_targets_data, eval_targets_scale)
            postprocess_targets(eval_targets_pred, eval_targets_scale)
        elif dictarray_is_not_none(targets):
            postprocess_targets(eval_targets_data, targets_scale)
            postprocess_targets(eval_targets_pred, targets_scale)
        elif dictarray_is_not_none(targets_noise):
            postprocess_targets(eval_targets_data, targets_noise_scale)
            postprocess_targets(eval_targets_pred, targets_noise_scale)
        else:
            raise NotImplementedError()

        # ----------------------------
        # Metrics dump (postprocessed units)
        # ----------------------------
        save_dir = params["runconfig"]["save_dir"]
        n_targets = None
        for _k, _arr in eval_targets_data.items():
            if getattr(_arr, "size", 0) > 0:
                n_targets = int(_arr.shape[1])
                break
        if n_targets is None:
            raise RuntimeError("No evaluation targets available to compute metrics.")

        cfg_target_names = params.get("data", {}).get("target_names")
        if cfg_target_names is None:
            target_names = [f"theta_{i+1}" for i in range(n_targets)]
        else:
            target_names = [str(x) for x in cfg_target_names]
            if len(target_names) != n_targets:
                raise ValueError(
                    "data.target_names length mismatch: "
                    f"len(target_names)={len(target_names)} but n_targets={n_targets}"
                )

        eval_mse, eval_mae, eval_r2, eval_per_target = eval_data_vs_pred(
            eval_targets_data, eval_targets_pred, target_names=target_names
        )

        import json, csv
        def _to_py(x):
            try:
                return float(x)
            except Exception:
                return x

        metrics_payload = {
            "metadata": {
                "dataset_layout": params.get("data", {}).get("dataset_layout"),
                "data_prefix": params.get("data", {}).get("data_prefix"),
                "curr": params.get("data", {}).get("curr"),
                "target_names": target_names,
            },
            "mse": {k: _to_py(v) for k, v in eval_mse.items()},
            "mae": {k: _to_py(v) for k, v in eval_mae.items()},
            "r2":  {k: _to_py(v) for k, v in eval_r2.items()},
        }
        with open(os.path.join(save_dir, "metrics_summary.json"), "w") as f:
            json.dump(metrics_payload, f, indent=2)

        with open(os.path.join(save_dir, "metrics_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "mse", "mae", "r2"])
            for split in sorted(eval_mse.keys()):
                w.writerow([split, eval_mse[split], eval_mae[split], eval_r2[split]])

        with open(os.path.join(save_dir, "target_names.json"), "w") as f:
            json.dump({"target_names": target_names}, f, indent=2)

        with open(os.path.join(save_dir, "metrics_per_target.json"), "w") as f:
            json.dump(
                {
                    "metadata": metrics_payload["metadata"],
                    "per_target": eval_per_target,
                },
                f,
                indent=2,
            )

        with open(os.path.join(save_dir, "metrics_per_target.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "target_index", "target_name", "mse", "rmse", "mae", "r2", "mape_percent", "nmae_range"])
            for split in sorted(eval_per_target.keys()):
                for row in eval_per_target[split]:
                    w.writerow([
                        split,
                        row["target_index"],
                        row["target_name"],
                        row["mse"],
                        row["rmse"],
                        row["mae"],
                        row["r2"],
                        row["mape_percent"],
                        row["nmae_range"],
                    ])

        # ----------------------------
        # Per-sample predictions dump
        # ----------------------------
        save_predictions = params["runconfig"].get("save_predictions", "test")
        if save_predictions != "none":
            keys = list(eval_targets_data.keys()) if save_predictions == "all" else ["test"]
            out = {}
            for k in keys:
                if k in eval_targets_data and k in eval_targets_pred:
                    if getattr(eval_targets_data[k], "size", 0) > 0:
                        out[f"{k}_true"] = eval_targets_data[k]
                        out[f"{k}_pred"] = eval_targets_pred[k]
            if out:
                np.savez_compressed(os.path.join(save_dir, "predictions.npz"), **out)
                with open(os.path.join(save_dir, "predictions_schema.json"), "w") as f:
                    json.dump(
                        {
                            "file": "predictions.npz",
                            "arrays": sorted(list(out.keys())),
                            "description": "Per-sample arrays in original parameter units. Keys are <split>_(true|pred).",
                        },
                        f,
                        indent=2,
                    )

        # ----------------------------
        # Graphs (PDF, as in the original codebase)
        # ----------------------------
        try:
            import matplotlib.pyplot as plt

            if save_diagnostic_plots and "test" in eval_targets_data and getattr(eval_targets_data["test"], "size", 0) > 0:
                y_true = eval_targets_data["test"]
                y_pred = eval_targets_pred["test"]
                err = y_pred - y_true
                npar = y_true.shape[1]

                for j in range(npar):
                    plt.figure()
                    plt.scatter(y_true[:, j], y_pred[:, j], s=6)
                    plt.xlabel(f"theta_true[{j}]")
                    plt.ylabel(f"theta_pred[{j}]")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"scatter_true_vs_pred_param{j}.pdf"), dpi=300)
                    plt.close()

                for j in range(npar):
                    plt.figure()
                    plt.hist(err[:, j], bins=80, density=True)
                    plt.xlabel(f"theta_pred[{j}] - theta_true[{j}]")
                    plt.ylabel("density")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"hist_error_param{j}.pdf"), dpi=300)
                    plt.close()

        except Exception as e:
            logging_get_logger("predict").warning(f"Plotting skipped: {e}")

        print('</predict>', flush=True)

        #
        # Output
        #

        logger.info(f"Runtime - train [sec]: {time_train}")
        logger.info(f"Runtime - eval [sec]:  {time_eval}")

        if (train_dlog is not None) and (0 < time_train):
            n_epoch   = params['training']['epochs']
            n_steps   = params['training']['epochs'] * (params['data']['Ntrain']//params['data']['train_batch_size'])
            n_samples = params['data']['train_batch_size']
            logger.info(f"Runtime statistics - train - #epochs:          {n_epoch}")
            logger.info(f"Runtime statistics - train - #steps:           {n_steps}")
            logger.info(f"Runtime statistics - train - #samples (total): {n_steps*n_samples}")
            logger.info(f"Runtime statistics - train - avg. steps/sec:   {n_steps/time_train}")
            logger.info(f"Runtime statistics - train - avg. samples/sec: {n_steps*n_samples/time_train}")

            # plot loss
            path = os.path.join(self_dir, params['runconfig']['save_dir'], 'loss')
            plot_loss(train_dlog['loss_mean'], path, 'Training loss', params['training']['epochs'],
                      loss_std=train_dlog['loss_std'], x_offset=1, y_scale='log',
                      save_plot=save_diagnostic_plots)

        if 0 < time_eval:
            n_samples = (params['data']['Ntest']//params['data']['eval_batch_size']) * params['data']['eval_batch_size']
            logger.info(f"Runtime statistics - eval  - #samples:         {n_samples}")
            logger.info(f"Runtime statistics - eval  - avg. samples/sec: {n_samples/time_eval}")

        if params['runconfig']['show_plots']:
            plt.show()

    finally:
        if dist_on and dist.is_initialized():
            dist.destroy_process_group()



###############################################################################

def predict(net, eval_dataloader, params, device, input_transform_fn=None):
    net.eval()
    # get network predictions
    data = dict()
    pred = dict()
    with torch.no_grad():
        for key in eval_dataloader.keys():
            d_list = list()
            p_list = list()
            for x, yd in tqdm(eval_dataloader[key], desc=key):
                x = x.to(device)
                if input_transform_fn is not None:
                    x = input_transform_fn(x)
                yp = net(x)
                d_list.append(yd.cpu().numpy())
                p_list.append(yp.cpu().numpy())
            data[key] = np.concatenate(d_list, axis=0)
            pred[key] = np.concatenate(p_list, axis=0)
    # return predictions and (true) data
    return pred, data

def eval_data_vs_pred(data, pred, target_names=None):
    eval_mse = dict()
    eval_mae = dict()
    eval_r2  = dict()
    eval_per_target = dict()
    for key in data.keys():
        data_ = data[key]
        pred_ = pred[key]
        n_params = int(data_.shape[1])
        names = [f"theta_{i+1}" for i in range(n_params)] if target_names is None else target_names

        mse_i = [metrics.mean_squared_error(data_[:, i], pred_[:, i]) for i in range(n_params)]
        mae_i = [metrics.mean_absolute_error(data_[:, i], pred_[:, i]) for i in range(n_params)]
        r2_i = [metrics.r2_score(data_[:, i], pred_[:, i]) for i in range(n_params)]
        rmse_i = [float(np.sqrt(v)) for v in mse_i]

        per_rows = []
        for i in range(n_params):
            y_true = data_[:, i]
            y_pred = pred_[:, i]
            abs_true = np.abs(y_true)
            denom = np.where(abs_true > 1.0e-12, abs_true, np.nan)
            ape = np.abs(y_pred - y_true) / denom
            mape_percent = float(np.nanmean(ape) * 100.0)
            y_range = float(np.max(y_true) - np.min(y_true))
            nmae_range = float(mae_i[i] / y_range) if y_range > 1.0e-12 else float("nan")
            per_rows.append(
                {
                    "target_index": int(i),
                    "target_name": str(names[i]),
                    "mse": float(mse_i[i]),
                    "rmse": float(rmse_i[i]),
                    "mae": float(mae_i[i]),
                    "r2": float(r2_i[i]),
                    "mape_percent": float(mape_percent),
                    "nmae_range": float(nmae_range),
                }
            )

        eval_per_target[key] = per_rows
        eval_mse[key] = metrics.mean_squared_error(data_, pred_)
        eval_mae[key] = metrics.mean_absolute_error(data_, pred_)
        eval_r2[key] = metrics.r2_score(data_, pred_)

    return eval_mse, eval_mae, eval_r2, eval_per_target

###############################################################################

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--params",
        default="./configs/params_dnn.yaml",
        help="Path to .yaml file with parameters",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["train", "predict", "eval", "train_eval", "train_predict", "train_profile"],
        default="train_eval",
        help=(
            "Can train, predict, eval, and combine train_eval (default). "
            "eval runs on available checkpoints. "
            "train_eval runs train, predict, and eval. "
            "train_profile runs profiling of a few training steps."
        ),
    )

    # Data overrides (needed for extracted tar paths)
    parser.add_argument("--data_dir", default=None, help="Override data.data_dir")
    parser.add_argument("--data_prefix", default=None, help="Override data.data_prefix")
    parser.add_argument("--curr", default=None, help="Value for {curr} in file_names templates")

    # Output location: base + SLURM_JOB_ID
    parser.add_argument(
        "--save_dir_base",
        default="/projects/neuro-collab/data/runs",
        help="Base output directory; run directory appends SLURM_JOB_ID (or a timestamp if unset).",
    )
    parser.add_argument(
        "--load_dir",
        default=None,
        help="Optional checkpoint path override for eval/predict resume.",
    )

    # Optional extra evaluation datasets, repeatable:
    #   --eval_set name:/abs/path/to/extracted_dataset
    parser.add_argument(
        "--eval_set",
        action="append",
        default=None,
        help="Additional evaluation set as name:/abs/path, repeatable.",
    )

    # Persist per-sample predictions
    parser.add_argument(
        "--save_predictions",
        choices=["none", "test", "all"],
        default="test",
        help="Persist per-sample predictions as .npz (default: test).",
    )
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])

    params = load_parameters(args.params)

    # existing: update runconfig from args
    update_parameters_from_args(params["runconfig"], args)

    # data overrides (extracted dataset root, curr template var)
    if getattr(args, "data_dir", None) is not None:
        params["data"]["data_dir"] = args.data_dir
    if getattr(args, "data_prefix", None) is not None:
        params["data"]["data_prefix"] = args.data_prefix
    if getattr(args, "curr", None) is not None:
        params["data"]["curr"] = args.curr
    if getattr(args, "load_dir", None) is not None:
        params["runconfig"]["load_dir"] = args.load_dir

    # persist preference for prediction saving
    if getattr(args, "save_predictions", None) is not None:
        params["runconfig"]["save_predictions"] = args.save_predictions

    # resolve run directory from SLURM_JOB_ID
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        job_id = time.strftime("manual_%Y%m%d_%H%M%S")

    save_dir_base = getattr(args, "save_dir_base", "/projects/neuro-collab/data/runs")
    params["runconfig"]["save_dir"] = os.path.join(save_dir_base, str(job_id))

    os.makedirs(params["runconfig"]["save_dir"], exist_ok=True)

    # save params for reproducibility
    save_parameters(params, save_dir=params["runconfig"]["save_dir"])

    run(args, params)


if __name__ == "__main__":
    main()
