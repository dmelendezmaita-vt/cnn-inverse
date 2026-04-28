"""
Handling of data.
"""

import hashlib, inspect, io, json, logging, pathlib, os, shutil, sys, random, tarfile, time, warnings
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import Mode

###############################################################################

def dictarray_empty():
    return {'train': None, 'validate': None, 'test': None}

def dictarray_set(arr_train, arr_validate, arr_test):
    return {'train': arr_train, 'validate': arr_validate, 'test': arr_test}

def dictarray_is_none(arr):
    return arr             is None or \
           arr['train']    is None or \
           arr['validate'] is None or \
           arr['test']     is None

def dictarray_is_not_none(arr):
    return not dictarray_is_none(arr)


class IndexedArray:
    """
    Lightweight row-indexed view over a base numpy array.

    The view keeps one backing array plus explicit row indices so downstream
    code can avoid eagerly materializing train/validate/test copies.
    """

    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1:
            raise ValueError(f"IndexedArray indices must be 1D, got shape={self.indices.shape}")

    @property
    def shape(self):
        return (int(self.indices.shape[0]), *self.base.shape[1:])

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return n

    def __len__(self):
        return int(self.indices.shape[0])

    def materialize(self):
        return np.asarray(self.base[self.indices, ...])

    def __array__(self, dtype=None):
        arr = self.materialize()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.materialize()[idx]
        return self.base[self.indices[idx], ...]


def dictarray_uses_indexed_arrays(arr):
    if arr is None:
        return False
    return any(isinstance(arr.get(key), IndexedArray) for key in ("train", "validate", "test"))


def _dist_rank_info():
    dist_init = False
    rank = 0
    try:
        import torch.distributed as dist
        dist_init = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if dist_init else 0
    except Exception:
        dist_init = False
        rank = 0
    return dist_init, rank


def _resolve_features_scale_cache_path(data_params, array_name='features'):
    if array_name != 'features':
        return None
    if not bool(data_params.get('features_scale_cache_enabled', False)):
        return None

    explicit = data_params.get('features_scale_cache_path')
    if explicit:
        return pathlib.Path(str(explicit))

    data_dir = pathlib.Path(data_params['data_dir'])
    if not data_dir.exists() or not data_dir.is_dir():
        return None

    payload = {
        'version': 1,
        'array_name': array_name,
        'dataset_layout': data_params.get('dataset_layout'),
        'data_dir': str(data_dir),
        'data_prefix': data_params.get('data_prefix'),
        'curr': data_params.get('curr'),
        'features_type': data_params.get('features_type'),
        'features_use_channels_range': data_params.get('features_use_channels_range'),
        'features_use_length_range': data_params.get('features_use_length_range'),
        'features_normalize': data_params.get('features_normalize', False),
        'Ntrain': data_params.get('Ntrain'),
        'Nvalidate': data_params.get('Nvalidate'),
        'Ntest': data_params.get('Ntest'),
        'split_strategy': data_params.get('split_strategy', 'sequential'),
        'split_seed': data_params.get('split_seed', data_params.get('random_seed', 0)),
        'legacy_validate_test_overlap': bool(data_params.get('legacy_validate_test_overlap', False)),
        'theta_filter': data_params.get('theta_filter'),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]
    return data_dir / '.scale_cache' / f'{array_name}_scale_{digest}.npz'


def _load_features_scale_cache(cache_path: pathlib.Path):
    with np.load(cache_path, allow_pickle=False) as cached:
        shift = np.array(cached['shift'])
        mult = np.array(cached['mult'])
    return {'shift': shift, 'mult': mult}


def _save_features_scale_cache(cache_path: pathlib.Path, scale):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.parent / f".{cache_path.stem}.{os.getpid()}.tmp.npz"
    with tmp_path.open("wb") as f:
        np.savez(f, shift=np.asarray(scale['shift']), mult=np.asarray(scale['mult']))
    os.replace(tmp_path, cache_path)


def _resolve_split_array_cache_dir(data_params):
    if not bool(data_params.get('split_array_cache_enabled', False)):
        return None

    explicit = data_params.get('split_array_cache_dir')
    if explicit:
        return pathlib.Path(str(explicit))

    data_dir = pathlib.Path(data_params['data_dir'])
    if not data_dir.exists() or not data_dir.is_dir():
        return None

    split_strategy = data_params.get('split_strategy', 'sequential')
    split_seed = data_params.get('split_seed')
    if split_strategy != 'sequential' and split_seed is None:
        split_seed = data_params.get('random_seed', 0)

    payload = {
        'version': 2,
        'dataset_layout': data_params.get('dataset_layout'),
        'data_dir': str(data_dir),
        'data_prefix': data_params.get('data_prefix'),
        'curr': data_params.get('curr'),
        'features_type': data_params.get('features_type'),
        'targets_type': data_params.get('targets_type'),
        'features_use_channels_range': data_params.get('features_use_channels_range'),
        'features_use_length_range': data_params.get('features_use_length_range'),
        'Ntrain': data_params.get('Ntrain'),
        'Nvalidate': data_params.get('Nvalidate'),
        'Ntest': data_params.get('Ntest'),
        'split_strategy': split_strategy,
        'split_seed': split_seed,
        'legacy_validate_test_overlap': bool(data_params.get('legacy_validate_test_overlap', False)),
        'theta_filter': data_params.get('theta_filter'),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]
    cache_dir = data_dir / '.split_array_cache' / digest
    return _maybe_stage_split_array_cache_dir(cache_dir)


def _split_array_cache_files(cache_dir: pathlib.Path, array_name: str):
    return {
        'train': cache_dir / f'{array_name}_train.npy',
        'validate': cache_dir / f'{array_name}_validate.npy',
        'test': cache_dir / f'{array_name}_test.npy',
    }


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, '')
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _stage_file_atomic(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f'.{dst.name}.{os.getpid()}.tmp'
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _maybe_stage_split_array_cache_dir(cache_dir: pathlib.Path) -> pathlib.Path:
    if not _env_truthy('NC_STAGE_SPLIT_CACHE_TO_TMPDIR'):
        return cache_dir
    stage_root_raw = os.environ.get('NC_LOCAL_STAGE_ROOT') or os.environ.get('TMPDIR') or ''
    if not stage_root_raw:
        return cache_dir
    stage_root = pathlib.Path(stage_root_raw)
    if not stage_root.exists() or not stage_root.is_dir():
        return cache_dir

    source_files = []
    for array_name in ('features', 'targets'):
        for split, path in _split_array_cache_files(cache_dir, array_name).items():
            source_files.append((f'{array_name}_{split}', path))
    if not all(path.exists() for _, path in source_files):
        return cache_dir

    staged_dir = stage_root / 'neuro_collab_split_array_cache' / cache_dir.name
    staged_files = {name: staged_dir / path.name for name, path in source_files}
    source_lookup = dict(source_files)
    if all(path.exists() and path.stat().st_size == source_lookup[name].stat().st_size for name, path in staged_files.items()):
        return staged_dir

    staged_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = staged_dir.parent / f'.{cache_dir.name}.lock'
    lock_fd = None
    while True:
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if all(path.exists() and path.stat().st_size == source_lookup[name].stat().st_size for name, path in staged_files.items()):
                return staged_dir
            time.sleep(1.0)

    try:
        if all(path.exists() and path.stat().st_size == source_lookup[name].stat().st_size for name, path in staged_files.items()):
            return staged_dir
        staged_dir.mkdir(parents=True, exist_ok=True)
        for name, src in source_files:
            dst = staged_files[name]
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            _stage_file_atomic(src, dst)
        return staged_dir
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _load_split_array_cache(cache_dir: pathlib.Path, array_name: str, mmap_mode=None):
    files = _split_array_cache_files(cache_dir, array_name)
    if not all(path.exists() for path in files.values()):
        return None
    return {
        split: np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
        for split, path in files.items()
    }


def _save_split_array_cache(cache_dir: pathlib.Path, array_name: str, arrays):
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = _split_array_cache_files(cache_dir, array_name)
    tmp_files = {}
    try:
        for split, path in files.items():
            tmp_path = cache_dir / f'.{path.stem}.{os.getpid()}.tmp.npy'
            with tmp_path.open('wb') as f:
                np.save(f, np.asarray(arrays[split]))
            tmp_files[split] = tmp_path
        for split, path in files.items():
            os.replace(tmp_files[split], path)
    finally:
        for tmp_path in tmp_files.values():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

###############################################################################

def _load_memmap(data_file, cols_num, dtype=np.float32):
    data_points = cols_num
    data_file_size = os.path.getsize(data_file)
    data_rows = data_file_size // (data_points * 4)
    return np.memmap(
        data_file,
        dtype=dtype,
        mode='r',
        shape=(data_rows, data_points)
    )


def _validate_cols_num(array, cols_num, context):
    """
    Validate that the final dimension matches the configured column count.
    This catches accidental schema drift early (especially for tar/raw datasets).
    """
    if cols_num is None:
        return
    if array.ndim < 2:
        raise ValueError(
            f"{context}: expected at least 2D array, got shape={getattr(array, 'shape', None)}"
        )
    expected = int(cols_num)
    actual = int(array.shape[-1])
    if actual != expected:
        raise ValueError(
            f"{context}: expected last-dimension cols_num={expected}, got {actual} "
            f"(shape={array.shape})"
        )

def _load_array(
    data_file,
    cols_num=None,
    dtype=np.float32,
    expand_dims_axis=None,
    mmap_mode=None,
):
    """
    Loads numpy arrays from disk.
    - Uses np.load for .npy files.
    - If np.load fails (non-npy raw binary), falls back to manual memmap loader.
    - If mmap_mode is not None, uses np.load(..., mmap_mode=...).
    """
    try:
        if mmap_mode is None:
            array = np.load(data_file)
        else:
            array = np.load(data_file, mmap_mode=mmap_mode)
    except (ValueError, OSError) as e:
        # Raw binary fallback for datasets stored as float32 blobs with .npy suffix.
        if cols_num is None or dtype is None:
            raise ValueError(
                f"Failed to load '{data_file}' via np.load and raw-binary fallback cannot run "
                f"because cols_num/dtype is missing (cols_num={cols_num}, dtype={dtype}). "
                f"Set data.features_cols_num / data.targets_cols_num (and *_noise_cols_num if needed). "
                f"Original error: {e}"
            ) from e
        array = _load_memmap(data_file, cols_num=cols_num, dtype=dtype)

    assert 2 <= array.ndim
    if expand_dims_axis is not None and array.ndim < 3:
        array = np.expand_dims(array, axis=expand_dims_axis)
    _validate_cols_num(array, cols_num, context=f"_load_array('{data_file}')")
    return array


def _safe_format(template: str, mapping: dict) -> str:
    class _SafeDict(dict):
        def __missing__(self, k):
            return "{" + k + "}"
    if template is None:
        return None
    return template.format_map(_SafeDict(mapping))


def _resolve_tar_member_name(tf, member_name: str, data_prefix=None):
    normalized = str(member_name).lstrip("./")
    if not normalized:
        return None

    candidates = []
    seen = set()

    def _add(cand):
        if cand not in seen:
            seen.add(cand)
            candidates.append(cand)

    # direct and dot-prefixed
    _add(normalized)
    _add(f"./{normalized}")

    # explicit data_prefix (preferred for dataset roots like concatenated_data/)
    if data_prefix:
        root = str(data_prefix).strip("/.")
        if root:
            _add(f"{root}/{normalized}")
            _add(f"./{root}/{normalized}")

    # legacy compatibility
    _add(f"reduced_data/{normalized}")
    _add(f"./reduced_data/{normalized}")

    # Generic root-folder fallback: if tar contains a top-level directory, try root/member.
    # This keeps tar loading robust when archives are packaged with a dataset root folder.
    roots = set()
    try:
        for name in tf.getnames():
            if not name:
                continue
            clean = str(name).lstrip("./")
            if "/" in clean:
                roots.add(clean.split("/", 1)[0])
    except Exception:
        roots = set()
    for root in sorted(roots):
        _add(f"{root}/{normalized}")
        _add(f"./{root}/{normalized}")

    for cand in candidates:
        try:
            tf.getmember(cand)
            return cand
        except KeyError:
            continue
    return None


def _load_array_from_tar(
    tar_path,
    member_name,
    data_prefix=None,
    cols_num=None,
    dtype=np.float32,
    expand_dims_axis=None,
    mmap_mode=None,
):
    """
    Load a numpy array directly from a tar member without extracting files.
    """
    with tarfile.open(tar_path, mode="r:*") as tf:
        resolved = _resolve_tar_member_name(tf, str(member_name), data_prefix=data_prefix)
        if resolved is None:
            raise FileNotFoundError(f"Tar member not found: {member_name} in {tar_path}")
        fh = tf.extractfile(resolved)
        if fh is None:
            raise FileNotFoundError(f"Unable to open tar member: {resolved} in {tar_path}")

        # In-memory streams do not support mmap_mode semantics.
        if mmap_mode is not None:
            mmap_mode = None

        raw = fh.read()

        try:
            # np.load over tar extract streams may fail because fileno() is unavailable.
            # Route through BytesIO to support standard .npy payloads inside tar archives.
            array = np.load(io.BytesIO(raw))
        except (ValueError, OSError, EOFError, AttributeError) as e:
            # Raw-binary fallback for .npy-labeled float blobs.
            if cols_num is None or dtype is None:
                raise ValueError(
                    f"Failed to load tar member '{member_name}' via np.load and raw-binary fallback "
                    f"cannot run because cols_num/dtype is missing (cols_num={cols_num}, dtype={dtype}). "
                    f"Original error: {e}"
                ) from e
            if cols_num <= 0:
                raise ValueError(f"cols_num must be > 0 for raw-binary fallback, got {cols_num}")
            arr = np.frombuffer(raw, dtype=dtype)
            if arr.size % cols_num != 0:
                raise ValueError(
                    f"Raw tar member '{member_name}' cannot be reshaped with cols_num={cols_num}; "
                    f"elements={arr.size}"
                ) from e
            array = arr.reshape((-1, cols_num))

    assert 2 <= array.ndim
    if expand_dims_axis is not None and array.ndim < 3:
        array = np.expand_dims(array, axis=expand_dims_axis)
    _validate_cols_num(array, cols_num, context=f"_load_array_from_tar('{tar_path}:{member_name}')")
    return array


def _load_and_split_arrays(data_params, logger=None):
    # set up logger
    if logger is None:
        logger = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")

    t_total_start = time.perf_counter()
    timings_sec = {}

    def _timed(label, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        timings_sec[label] = timings_sec.get(label, 0.0) + (time.perf_counter() - t0)
        return out

    # options
    features_type = data_params['features_type'].casefold()
    targets_type  = data_params.get('targets_type', 'N/A').casefold()
    data_dir      = pathlib.Path(data_params['data_dir'])
    data_dir_str  = str(data_dir)
    tar_data_source = (
        data_dir.is_file()
        and (
            data_dir_str.endswith(".tar")
            or data_dir_str.endswith(".tar.gz")
            or data_dir_str.endswith(".tgz")
            or data_dir_str.endswith(".tar.bz2")
            or data_dir_str.endswith(".tar.xz")
        )
    )

    # explicit layout selector
    dataset_layout = data_params.get('dataset_layout')
    if dataset_layout is None:
        dataset_layout = 'legacy_2020_hardcoded' if '2020' in data_dir.name else 'paired_train_test'
    dataset_layout = dataset_layout.casefold()

    # file names (allow templates like {curr})
    file_names = data_params.get(
        'file_names',
        {
            'features'           : 'fhn_Ntrain20000_state_Nt2000_dt0.2.npy',
            'features_test'      : 'fhn_Ntest2000_state_Nt2000_dt0.2.npy',
            'features_stats'     : 'fhn_Ntrain20000_state_stats.npy',
            'features_stats_test': 'fhn_Ntest2000_state_stats.npy',
            'targets'            : 'fhn_Ntrain20000_param.npy',
            'targets_test'       : 'fhn_Ntest2000_param.npy',
            'features_noise'     : 'ar1_Ntrain20000_state_Nt2000_dt0.2.npy',
            'features_noise_test': 'ar1_Ntest2000_state_Nt2000_dt0.2.npy',
            'targets_noise'      : 'ar1_Ntrain20000_param.npy',
            'targets_noise_test' : 'ar1_Ntest2000_param.npy',
        }
    )

    # Allow templates in file_names, e.g. "y/{data_prefix}_{curr}_curr.npy"
    tmpl_vars = {
        k: v for k, v in data_params.items()
        if isinstance(v, (str, int, float, bool)) and v is not None
    }
    file_names = {
        k: (v.format_map(tmpl_vars) if isinstance(v, str) and ("{" in v and "}" in v) else v)
        for k, v in file_names.items()
    }


    lazy_split_views_requested = bool(data_params.get("lazy_split_views", False))

    # memmap / mmap settings
    use_mmap = bool(data_params.get('use_mmap', False))
    mmap_mode = "c" if (use_mmap and lazy_split_views_requested) else ("r" if use_mmap else None)
    if tar_data_source and mmap_mode is not None:
        logger.warning("data_dir points to tar archive; disabling mmap_mode because tar-stream loading is in-memory")
        mmap_mode = None
    data_source_kind = "tar_archive" if tar_data_source else "filesystem_dir"
    logger.info(
        "data load config: dataset_layout=%s data_source=%s use_mmap_requested=%s use_mmap_effective=%s data_dir=%s",
        dataset_layout,
        data_source_kind,
        use_mmap,
        mmap_mode is not None,
        data_dir,
    )

    features_use_channels_range = data_params.get('features_use_channels_range')
    features_use_length_range   = data_params.get('features_use_length_range')
    features_cols_num           = data_params.get('features_cols_num')
    targets_cols_num            = data_params.get('targets_cols_num')
    features_noise_cols_num     = data_params.get('features_noise_cols_num')
    targets_noise_cols_num      = data_params.get('targets_noise_cols_num')

    Ntrain    = data_params['Ntrain']
    Nvalidate = data_params.get('Nvalidate', 0) or 0
    Ntest     = data_params.get('Ntest')
    legacy_validate_test_overlap = bool(data_params.get("legacy_validate_test_overlap", False))

    def _apply_feature_slices(arr):
        if arr is None:
            return None
        if features_use_channels_range is not None:
            assert 2 == len(features_use_channels_range)
            start, stop = features_use_channels_range
            arr = arr[:, start:stop, :]
        if features_use_length_range is not None:
            assert 2 == len(features_use_length_range)
            start, stop = features_use_length_range
            arr = arr[:, :, start:stop]
        return arr

    def _p(name_key: str):
        if name_key not in file_names:
            raise KeyError(f"Missing file_names['{name_key}'] for dataset_layout={dataset_layout}")
        rel = _safe_format(str(file_names[name_key]), data_params)
        if tar_data_source:
            return rel
        return data_dir / rel

    def _load_array_maybe_tar(
        data_ref,
        cols_num=None,
        dtype=np.float32,
        expand_dims_axis=None,
        mmap_mode=None,
    ):
        if tar_data_source:
            return _load_array_from_tar(
                data_dir,
                data_ref,
                data_prefix=data_params.get('data_prefix'),
                cols_num=cols_num,
                dtype=dtype,
                expand_dims_axis=expand_dims_axis,
                mmap_mode=mmap_mode,
            )
        return _load_array(
            data_ref,
            cols_num=cols_num,
            dtype=dtype,
            expand_dims_axis=expand_dims_axis,
            mmap_mode=mmap_mode,
        )

    # -------------------------------------------------------------------------
    # Load arrays depending on layout
    # -------------------------------------------------------------------------

    if dataset_layout == 'legacy_2020_hardcoded':
        # default
        if Ntest is None:
            Ntest = 2000

        # FEATURES (FIXED SPLIT BUG: use features_all)
        if features_type in ['time', 'time_noise']:
            features_all = _timed(
                "load.features_all",
                lambda: np.expand_dims(np.load(data_dir/'fhn_T200_samplePrior_state0.npy', mmap_mode=mmap_mode), axis=1),
            )
            features_     = features_all[:-Ntest, ...]
            features_test = features_all[-Ntest:, ...]
            features_     = _timed("slice.features_pool", _apply_feature_slices, features_)
            features_test = _timed("slice.features_test", _apply_feature_slices, features_test)
            logger.debug(f"{features_type=}, {features_.shape=}, {features_test.shape=}")

        elif features_type in ['ode_stats', 'rate_duration']:
            rate = _timed("load.features_rate", np.load, data_dir/'fhn_T200_samplePrior_spikeRate.npy', mmap_mode=mmap_mode)
            duration = _timed("load.features_duration", np.load, data_dir/'fhn_T200_samplePrior_spikeDuration.npy', mmap_mode=mmap_mode)
            features_all = _timed("build.features_all", lambda: np.expand_dims(np.stack((rate, duration), axis=1), axis=1))
            features_     = features_all[:-Ntest, ...]
            features_test = features_all[-Ntest:, ...]
            logger.debug(f"{features_type=}, {features_.shape=}, {features_test.shape=}")

        elif features_type == 'noise':
            features_ = features_test = None
        else:
            raise ValueError(f"Unknown {features_type=}")

        # TARGETS (FIXED SPLIT BUG)
        if targets_type in ['ode', 'ode_noise']:
            targets_all = _timed("load.targets_all", np.load, data_dir/'fhn_T200_samplePrior_theta.npy', mmap_mode=mmap_mode)
            targets_     = targets_all[:-Ntest, ...]
            targets_test = targets_all[-Ntest:, ...]
            logger.debug(f"{targets_type=}, {targets_.shape=}, {targets_test.shape=}")
        elif targets_type in ['noise', 'n/a']:
            targets_ = targets_test = None
        else:
            raise ValueError(f"Unknown {targets_type=}")

        # FEATURES NOISE (FIXED SPLIT BUG)
        if features_type in ['time_noise', 'noise']:
            features_noise_all = _timed(
                "load.features_noise_all",
                lambda: np.expand_dims(
                    np.load(data_dir/'noise_correlated_Nt1000_Nsim10000_data.npy', mmap_mode=mmap_mode),
                    axis=1
                ),
            )
            features_noise_     = features_noise_all[:-Ntest, ...]
            features_noise_test = features_noise_all[-Ntest:, ...]
            features_noise_     = _timed("slice.features_noise_pool", _apply_feature_slices, features_noise_)
            features_noise_test = _timed("slice.features_noise_test", _apply_feature_slices, features_noise_test)
            logger.debug(f"{features_type=}, {features_noise_.shape=}, {features_noise_test.shape=}")
        elif features_type in ['time', 'ode_stats', 'rate_duration']:
            features_noise_ = features_noise_test = None
        else:
            raise ValueError(f"Unknown {features_type=}")

        # TARGETS NOISE (FIXED SPLIT BUG)
        if targets_type in ['noise', 'ode_noise']:
            noise_correl_all = _timed("load.targets_noise_correlation", np.load, data_dir/'noise_correlated_Nt1000_Nsim10000_correlation.npy', mmap_mode=mmap_mode)
            noise_stddev_all = _timed("load.targets_noise_stddev", np.load, data_dir/'noise_correlated_Nt1000_Nsim10000_stddev.npy', mmap_mode=mmap_mode)
            targets_noise_all = _timed("build.targets_noise_all", lambda: np.stack((noise_correl_all, noise_stddev_all), axis=1))
            targets_noise_     = targets_noise_all[:-Ntest, ...]
            targets_noise_test = targets_noise_all[-Ntest:, ...]
            logger.debug(f"{targets_type=}, {targets_noise_.shape=}, {targets_noise_test.shape=}")
        elif targets_type in ['ode', 'n/a']:
            targets_noise_ = targets_noise_test = None
        else:
            raise ValueError(f"Unknown {targets_type=}")

    elif dataset_layout == 'paired_train_test':
        # existing non-2020 behavior (plus mmap support)

        if features_type in ['time', 'time_noise']:
            features_ = _timed("load.features_pool", _load_array_maybe_tar, _p('features'), cols_num=features_cols_num, expand_dims_axis=1, mmap_mode=mmap_mode)
            features_test = _timed("load.features_test", _load_array_maybe_tar, _p('features_test'), cols_num=features_cols_num, expand_dims_axis=1, mmap_mode=mmap_mode)
            features_ = _timed("slice.features_pool", _apply_feature_slices, features_)
            features_test = _timed("slice.features_test", _apply_feature_slices, features_test)
            if Ntest is None:
                Ntest = features_test.shape[0]
            logger.debug(f"{features_type=}, {features_.shape=}, {features_test.shape=}")

        elif features_type in ['ode_stats', 'rate_duration']:
            features_ = _timed("load.features_pool", _load_array_maybe_tar, _p('features_stats'), cols_num=features_cols_num, expand_dims_axis=1, mmap_mode=mmap_mode)
            features_test = _timed("load.features_test", _load_array_maybe_tar, _p('features_stats_test'), cols_num=features_cols_num, expand_dims_axis=1, mmap_mode=mmap_mode)
            if Ntest is None:
                Ntest = features_test.shape[0]
            logger.debug(f"{features_type=}, {features_.shape=}, {features_test.shape=}")

        elif features_type == 'noise':
            features_ = features_test = None
        else:
            raise ValueError(f"Unknown {features_type=}")

        if targets_type in ['ode', 'ode_noise']:
            targets_ = _timed("load.targets_pool", _load_array_maybe_tar, _p('targets'), cols_num=targets_cols_num, mmap_mode=mmap_mode)
            targets_test = _timed("load.targets_test", _load_array_maybe_tar, _p('targets_test'), cols_num=targets_cols_num, mmap_mode=mmap_mode)
            logger.debug(f"{targets_type=}, {targets_.shape=}, {targets_test.shape=}")
        elif targets_type in ['noise', 'n/a']:
            targets_ = targets_test = None
        else:
            raise ValueError(f"Unknown {targets_type=}")

        if features_type in ['noise', 'time_noise']:
            features_noise_ = _timed("load.features_noise_pool", _load_array_maybe_tar, _p('features_noise'), cols_num=features_noise_cols_num, expand_dims_axis=1, mmap_mode=mmap_mode)
            features_noise_test = _timed("load.features_noise_test", _load_array_maybe_tar, _p('features_noise_test'), cols_num=features_noise_cols_num, expand_dims_axis=1, mmap_mode=mmap_mode)
            features_noise_ = _timed("slice.features_noise_pool", _apply_feature_slices, features_noise_)
            features_noise_test = _timed("slice.features_noise_test", _apply_feature_slices, features_noise_test)
            if Ntest is None:
                Ntest = features_noise_test.shape[0]
            logger.debug(f"{features_type=}, {features_noise_.shape=}, {features_noise_test.shape=}")
        else:
            features_noise_ = features_noise_test = None

        if targets_type in ['noise', 'ode_noise']:
            targets_noise_ = _timed("load.targets_noise_pool", _load_array_maybe_tar, _p('targets_noise'), cols_num=targets_noise_cols_num, mmap_mode=mmap_mode)
            targets_noise_test = _timed("load.targets_noise_test", _load_array_maybe_tar, _p('targets_noise_test'), cols_num=targets_noise_cols_num, mmap_mode=mmap_mode)
            logger.debug(f"{targets_type=}, {targets_noise_.shape=}, {targets_noise_test.shape=}")
        else:
            targets_noise_ = targets_noise_test = None

    elif dataset_layout == 'tar_singlefile_split':
        # NEW: one features file + one targets file, split into train-pool and test by last Ntest
        if Ntest is None:
            raise ValueError("For dataset_layout='tar_singlefile_split', you must set data.Ntest explicitly.")

        # FEATURES
        if features_type in ['time', 'time_noise']:
            features_all = _timed("load.features_all", _load_array_maybe_tar, _p('features'), cols_num=features_cols_num, expand_dims_axis=1, mmap_mode=mmap_mode)
            features_     = features_all[:-Ntest, ...]
            features_test = features_all[-Ntest:, ...]
            features_     = _timed("slice.features_pool", _apply_feature_slices, features_)
            features_test = _timed("slice.features_test", _apply_feature_slices, features_test)
            logger.debug(f"{features_type=}, {features_.shape=}, {features_test.shape=}")

        elif features_type in ['ode_stats', 'rate_duration']:
            features_all = _timed("load.features_all", _load_array_maybe_tar, _p('features_stats'), cols_num=features_cols_num, expand_dims_axis=1, mmap_mode=mmap_mode)
            features_     = features_all[:-Ntest, ...]
            features_test = features_all[-Ntest:, ...]
            logger.debug(f"{features_type=}, {features_.shape=}, {features_test.shape=}")

        elif features_type == 'noise':
            features_ = features_test = None
        else:
            raise ValueError(f"Unknown {features_type=}")

        # TARGETS
        if targets_type in ['ode', 'ode_noise']:
            targets_all = _timed("load.targets_all", _load_array_maybe_tar, _p('targets'), cols_num=targets_cols_num, mmap_mode=mmap_mode)
            targets_     = targets_all[:-Ntest, ...]
            targets_test = targets_all[-Ntest:, ...]
            logger.debug(f"{targets_type=}, {targets_.shape=}, {targets_test.shape=}")
        elif targets_type in ['noise', 'n/a']:
            targets_ = targets_test = None
        else:
            raise ValueError(f"Unknown {targets_type=}")

        # FEATURES NOISE (optional)
        if features_type in ['noise', 'time_noise']:
            features_noise_all = _timed("load.features_noise_all", _load_array_maybe_tar, _p('features_noise'), cols_num=features_noise_cols_num, expand_dims_axis=1, mmap_mode=mmap_mode)
            features_noise_     = features_noise_all[:-Ntest, ...]
            features_noise_test = features_noise_all[-Ntest:, ...]
            features_noise_     = _timed("slice.features_noise_pool", _apply_feature_slices, features_noise_)
            features_noise_test = _timed("slice.features_noise_test", _apply_feature_slices, features_noise_test)
            logger.debug(f"{features_type=}, {features_noise_.shape=}, {features_noise_test.shape=}")
        else:
            features_noise_ = features_noise_test = None

        # TARGETS NOISE (optional)
        if targets_type in ['noise', 'ode_noise']:
            targets_noise_all = _timed("load.targets_noise_all", _load_array_maybe_tar, _p('targets_noise'), cols_num=targets_noise_cols_num, mmap_mode=mmap_mode)
            targets_noise_     = targets_noise_all[:-Ntest, ...]
            targets_noise_test = targets_noise_all[-Ntest:, ...]
            logger.debug(f"{targets_type=}, {targets_noise_.shape=}, {targets_noise_test.shape=}")
        else:
            targets_noise_ = targets_noise_test = None

    else:
        raise ValueError(f"Unknown dataset_layout={dataset_layout}")

        # ----------------------------
    # Split arrays (index-based)
    # ----------------------------
    Nvalidate_eff = int(Nvalidate) if Nvalidate is not None else 0

    # Determine pool length from any available pool array
    ref_pool = None
    for cand in (targets_, targets_noise_, features_, features_noise_):
        if cand is not None and getattr(cand, "shape", None) is not None:
            ref_pool = cand
            break
    if ref_pool is None:
        raise ValueError("No pool arrays available for splitting.")
    n_pool = int(ref_pool.shape[0])
    idx_pool = np.arange(n_pool)

    # Optional truncation filter on targets_ (ODE parameters)
    theta_filter = data_params.get("theta_filter")
    if theta_filter is not None and targets_ is not None and targets_.size > 0:
        t0 = time.perf_counter()
        bounds = np.asarray(theta_filter.get("bounds"), dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("theta_filter.bounds must have shape (n_params, 2).")
        if bounds.shape[0] != targets_.shape[1]:
            raise ValueError(
                f"theta_filter.bounds has {bounds.shape[0]} params, but targets_ has {targets_.shape[1]}."
            )
        lo = bounds[:, 0]
        hi = bounds[:, 1]
        mask = np.ones(targets_.shape[0], dtype=bool)
        for j in range(bounds.shape[0]):
            mask &= (targets_[:, j] >= lo[j]) & (targets_[:, j] <= hi[j])

        rejected = int(np.sum(~mask))
        logger.info(f"theta_filter: rejected {rejected}/{targets_.shape[0]} pool samples")

        # apply to all pool arrays
        if features_ is not None and features_.size > 0:
            features_ = features_[mask, ...]
        if targets_ is not None and targets_.size > 0:
            targets_ = targets_[mask, ...]
        if features_noise_ is not None and features_noise_.size > 0:
            features_noise_ = features_noise_[mask, ...]
        if targets_noise_ is not None and targets_noise_.size > 0:
            targets_noise_ = targets_noise_[mask, ...]

        # refresh pool indices
        n_pool = int((targets_ if targets_ is not None and targets_.size > 0 else features_).shape[0])
        idx_pool = np.arange(n_pool)
        timings_sec["theta_filter"] = timings_sec.get("theta_filter", 0.0) + (time.perf_counter() - t0)

    # Split strategy: sequential | random | weighted
    split_strategy = str(data_params.get("split_strategy", "sequential")).casefold()
    split_seed = int(data_params.get("split_seed", data_params.get("random_seed", 0)))
    rng = np.random.default_rng(split_seed)

    if legacy_validate_test_overlap:
        if dataset_layout not in {"legacy_2020_hardcoded", "tar_singlefile_split"}:
            raise ValueError(
                "data.legacy_validate_test_overlap is only supported for "
                "dataset_layout in {'legacy_2020_hardcoded', 'tar_singlefile_split'}."
            )
        if split_strategy != "sequential":
            raise ValueError(
                "data.legacy_validate_test_overlap requires split_strategy='sequential'."
            )
        logger.warning(
            "legacy_validate_test_overlap enabled: emulating the original "
            "publication split behavior where test is drawn from the tail of the pool "
            "array instead of the true holdout tail."
        )

    if Ntrain + Nvalidate_eff > n_pool:
        raise ValueError(f"Requested Ntrain+Nvalidate ({Ntrain+Nvalidate_eff}) > available pool ({n_pool}).")

    t0 = time.perf_counter()
    if split_strategy == "sequential":
        idx_train = idx_pool[:Ntrain]
        idx_validate = idx_pool[-Nvalidate_eff:] if 0 < Nvalidate_eff else np.array([], dtype=int)

    elif split_strategy == "random":
        perm = idx_pool.copy()
        rng.shuffle(perm)
        idx_train = perm[:Ntrain]
        idx_validate = perm[Ntrain:Ntrain + Nvalidate_eff] if 0 < Nvalidate_eff else np.array([], dtype=int)

    elif split_strategy == "weighted":
        if targets_ is None or targets_.size == 0:
            raise ValueError("split_strategy='weighted' requires targets_ (ODE parameters).")
        wcfg = data_params.get("split_weights", {}) or {}
        method = str(wcfg.get("method", "gaussian")).casefold()

        if method == "gaussian":
            mu = np.asarray(wcfg["mu"], dtype=float)
            sigma = np.asarray(wcfg["sigma"], dtype=float)
            if mu.shape[0] != targets_.shape[1] or sigma.shape[0] != targets_.shape[1]:
                raise ValueError("split_weights.mu/sigma must match number of target parameters.")
            z = (targets_ - mu) / sigma
            logw = -0.5 * np.sum(z * z, axis=1)
            w = np.exp(logw - np.max(logw))
        else:
            raise ValueError(f"Unknown split_weights.method: {method}")

        w = w / np.sum(w)
        idx_train = rng.choice(idx_pool, size=Ntrain, replace=False, p=w)

        if 0 < Nvalidate_eff:
            remaining = np.setdiff1d(idx_pool, idx_train, assume_unique=False)
            idx_validate = rng.choice(remaining, size=Nvalidate_eff, replace=False)
        else:
            idx_validate = np.array([], dtype=int)

    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")
    timings_sec["split_indices"] = timings_sec.get("split_indices", 0.0) + (time.perf_counter() - t0)

    idx_test_pool = None
    if legacy_validate_test_overlap:
        idx_test_pool = idx_pool[-int(Ntest):] if 0 < int(Ntest) else np.array([], dtype=int)

    # Persist split indices for reproducibility if requested
    split_indices_path = data_params.get("split_indices_path")
    if split_indices_path:
        t0 = time.perf_counter()
        os.makedirs(os.path.dirname(split_indices_path), exist_ok=True)

        dist_init = False
        rank = 0
        try:
            import torch
            import torch.distributed as dist
            dist_init = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if dist_init else 0
        except Exception:
            dist_init = False
            rank = 0

        if (not dist_init) or (rank == 0):
            payload = {
                "idx_train": idx_train,
                "idx_validate": idx_validate,
            }
            if idx_test_pool is not None:
                payload["idx_test"] = idx_test_pool
            np.savez(split_indices_path, **payload)

        # Other ranks do not consume split_indices.npz during the active run, so
        # a global barrier here only adds startup jitter to data-load timing.
        if dist_init and bool(data_params.get("split_indices_barrier", False)):
            dist.barrier()
        timings_sec["persist_split_indices"] = timings_sec.get("persist_split_indices", 0.0) + (time.perf_counter() - t0)


    lazy_split_views = (
        lazy_split_views_requested
        and dataset_layout == 'tar_singlefile_split'
        and features_noise_ is None
        and targets_noise_ is None
        and features_type not in ['NOISE'.casefold(), 'TIME_NOISE'.casefold()]
        and targets_type not in ['NOISE'.casefold(), 'ODE_NOISE'.casefold()]
    )
    if lazy_split_views_requested and not lazy_split_views:
        logger.warning(
            "lazy_split_views requested but disabled for dataset_layout=%s features_type=%s targets_type=%s",
            dataset_layout,
            features_type,
            targets_type,
        )

    split_array_cache_dir = _resolve_split_array_cache_dir(data_params)
    split_array_cache_enabled = (
        split_array_cache_dir is not None
        and dataset_layout == 'tar_singlefile_split'
        and (not lazy_split_views)
        and (not tar_data_source)
        and features_noise_ is None
        and targets_noise_ is None
        and features_type in ['TIME'.casefold(), 'ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]
        and targets_type in ['ODE'.casefold()]
    )
    if bool(data_params.get("split_array_cache_enabled", False)) and not split_array_cache_enabled:
        logger.warning(
            "split_array_cache_enabled requested but disabled for dataset_layout=%s tar_data_source=%s features_type=%s targets_type=%s",
            dataset_layout,
            tar_data_source,
            features_type,
            targets_type,
        )
    split_array_cache_mmap_mode = "c" if (split_array_cache_enabled and use_mmap) else None

    def _split_array_view(base_arr, idx, use_views):
        if base_arr is None:
            return np.array([])
        if use_views:
            return IndexedArray(base_arr, idx)
        return base_arr[idx, ...]

    def _test_split_view(pool_arr, test_arr, use_views):
        if legacy_validate_test_overlap:
            if pool_arr is None or idx_test_pool is None or int(Ntest) <= 0:
                return np.array([])
            return _split_array_view(pool_arr, idx_test_pool, use_views)
        if test_arr is None or int(Ntest) <= 0:
            return np.array([])
        return _split_array_view(test_arr, np.arange(int(Ntest), dtype=np.int64), use_views)

    cached_features = None
    cached_targets = None
    if split_array_cache_enabled:
        t0 = time.perf_counter()
        cached_features = _load_split_array_cache(
            split_array_cache_dir,
            "features",
            mmap_mode=split_array_cache_mmap_mode,
        )
        timings_sec["load_split_array_cache_features"] = timings_sec.get("load_split_array_cache_features", 0.0) + (time.perf_counter() - t0)
        if cached_features is not None:
            logger.info("split array cache hit = %s (features)", split_array_cache_dir)
        t0 = time.perf_counter()
        cached_targets = _load_split_array_cache(
            split_array_cache_dir,
            "targets",
            mmap_mode=split_array_cache_mmap_mode,
        )
        timings_sec["load_split_array_cache_targets"] = timings_sec.get("load_split_array_cache_targets", 0.0) + (time.perf_counter() - t0)
        if cached_targets is not None:
            logger.info("split array cache hit = %s (targets)", split_array_cache_dir)

    # Build dictarrays
    t0 = time.perf_counter()
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
        if cached_features is not None:
            ft_train = cached_features['train']
            ft_validate = cached_features['validate']
            ft_test = cached_features['test']
        else:
            ft_train = _split_array_view(features_, idx_train, lazy_split_views) if (features_ is not None and 0 < Ntrain) else np.array([])
            ft_validate = _split_array_view(features_, idx_validate, lazy_split_views) if (features_ is not None and 0 < Nvalidate_eff) else np.array([])
            ft_test = _test_split_view(features_, features_test, lazy_split_views)
        features = dictarray_set(ft_train, ft_validate, ft_test)
    else:
        features = dictarray_empty()

    if targets_type in ['ODE'.casefold(), 'ODE_NOISE'.casefold()]:
        if cached_targets is not None:
            tg_train = cached_targets['train']
            tg_validate = cached_targets['validate']
            tg_test = cached_targets['test']
        else:
            tg_train = _split_array_view(targets_, idx_train, lazy_split_views) if (targets_ is not None and 0 < Ntrain) else np.array([])
            tg_validate = _split_array_view(targets_, idx_validate, lazy_split_views) if (targets_ is not None and 0 < Nvalidate_eff) else np.array([])
            tg_test = _test_split_view(targets_, targets_test, lazy_split_views)
        targets = dictarray_set(tg_train, tg_validate, tg_test)
    else:
        targets = dictarray_empty()

    if features_type in ['NOISE'.casefold(), 'TIME_NOISE'.casefold()]:
        ft_train    = features_noise_[idx_train, ...]    if (features_noise_ is not None and 0 < Ntrain) else np.array([])
        ft_validate = features_noise_[idx_validate, ...] if (features_noise_ is not None and 0 < Nvalidate_eff) else np.array([])
        if legacy_validate_test_overlap:
            ft_test = features_noise_[idx_test_pool, ...] if (features_noise_ is not None and idx_test_pool is not None and 0 < int(Ntest)) else np.array([])
        else:
            ft_test = features_noise_test[:Ntest, ...] if (features_noise_test is not None and 0 < int(Ntest)) else np.array([])
        features_noise = dictarray_set(ft_train, ft_validate, ft_test)
    else:
        features_noise = dictarray_empty()

    if targets_type in ['NOISE'.casefold(), 'ODE_NOISE'.casefold()]:
        tg_train    = targets_noise_[idx_train, ...]    if (targets_noise_ is not None and 0 < Ntrain) else np.array([])
        tg_validate = targets_noise_[idx_validate, ...] if (targets_noise_ is not None and 0 < Nvalidate_eff) else np.array([])
        if legacy_validate_test_overlap:
            tg_test = targets_noise_[idx_test_pool, ...] if (targets_noise_ is not None and idx_test_pool is not None and 0 < int(Ntest)) else np.array([])
        else:
            tg_test = targets_noise_test[:Ntest, ...] if (targets_noise_test is not None and 0 < int(Ntest)) else np.array([])
        targets_noise = dictarray_set(tg_train, tg_validate, tg_test)
    else:
        targets_noise = dictarray_empty()

    if split_array_cache_enabled and cached_features is None and dictarray_is_not_none(features):
        t_cache = time.perf_counter()
        dist_init, rank = _dist_rank_info()
        try:
            if (not dist_init) or (rank == 0):
                _save_split_array_cache(
                    split_array_cache_dir,
                    "features",
                    {
                        'train': features['train'],
                        'validate': features['validate'],
                        'test': features['test'],
                    },
                )
                logger.info("split array cache write = %s (features)", split_array_cache_dir)
        finally:
            if dist_init:
                import torch.distributed as dist
                dist.barrier()
        timings_sec["save_split_array_cache_features"] = timings_sec.get("save_split_array_cache_features", 0.0) + (time.perf_counter() - t_cache)

    if split_array_cache_enabled and cached_targets is None and dictarray_is_not_none(targets):
        t_cache = time.perf_counter()
        dist_init, rank = _dist_rank_info()
        try:
            if (not dist_init) or (rank == 0):
                _save_split_array_cache(
                    split_array_cache_dir,
                    "targets",
                    {
                        'train': targets['train'],
                        'validate': targets['validate'],
                        'test': targets['test'],
                    },
                )
                logger.info("split array cache write = %s (targets)", split_array_cache_dir)
        finally:
            if dist_init:
                import torch.distributed as dist
                dist.barrier()
        timings_sec["save_split_array_cache_targets"] = timings_sec.get("save_split_array_cache_targets", 0.0) + (time.perf_counter() - t_cache)
    timings_sec["build_dictarrays"] = timings_sec.get("build_dictarrays", 0.0) + (time.perf_counter() - t0)

    def _shape_or_none(arr):
        if arr is None:
            return None
        return tuple(int(x) for x in arr.shape)

    total_sec = time.perf_counter() - t_total_start
    timings_out = {k: round(v, 6) for k, v in sorted(timings_sec.items())}
    logger.info(
        "data split summary: strategy=%s pool_rows=%s train=%s validate=%s test=%s "
        "features_train=%s targets_train=%s features_test=%s targets_test=%s",
        split_strategy,
        n_pool,
        Ntrain,
        Nvalidate_eff,
        Ntest,
        _shape_or_none(features["train"]),
        _shape_or_none(targets["train"]),
        _shape_or_none(features["test"]),
        _shape_or_none(targets["test"]),
    )
    logger.info(
        "data load timings (sec): total=%.6f details=%s",
        total_sec,
        timings_out,
    )
    
    return features, targets, features_noise, targets_noise



def load_data(params, logger):
    data_params = params['data']

    # read data and split files
    features, targets, features_noise, targets_noise = _load_and_split_arrays(data_params, logger=logger)

    # print info
    if dictarray_is_not_none(features):
        for key in features.keys():
            logger.info(f"features['{key}']:\tshape {features[key].shape}, dtype {features[key].dtype}")
    if dictarray_is_not_none(targets):
        for key in targets.keys():
            logger.info(f"targets['{key}']: \tshape {targets[key].shape}, dtype {targets[key].dtype}")
    if dictarray_is_not_none(features_noise):
        for key in features_noise.keys():
            logger.info(f"features_noise['{key}']:\tshape {features_noise[key].shape}, dtype {features_noise[key].dtype}")
    if dictarray_is_not_none(targets_noise):
        for key in targets_noise.keys():
            logger.info(f"targets_noise['{key}']: \tshape {targets_noise[key].shape}, dtype {targets_noise[key].dtype}")

    # set feature sizes
    if "num_features" not in params["data"]:
        if dictarray_is_not_none(features):
            params['data']['num_features'] = list(features['train'].shape[1:])
            params['data'].setdefault('Ntest', features['test'].shape[0])
        elif dictarray_is_not_none(features_noise):
            params['data']['num_features'] = list(features_noise['train'].shape[1:])
            params['data'].setdefault('Ntest', features_noise['test'].shape[0])
        else:
            raise NotImplementedError()
        # set reduced feature sizes
        if params['data'].get('features_sub_length') and \
           params['data']['features_sub_length'] < params['data']['num_features'][-1]:
            params['data']['num_features'][-1] = params['data']['features_sub_length']
        if params['data'].get('features_sub_step') and 1 < params['data']['features_sub_step']:
            params['data']['num_features'][-1] = (
                params['data']['num_features'][-1] // params['data']['features_sub_step']
            )

    # set targets sizes
    if "num_targets" not in params["data"]:
        params['data']['num_targets'] = 0
        if dictarray_is_not_none(targets):
            assert dictarray_is_not_none(features)
            params['data']['num_targets'] += targets['train'].shape[1]
        if dictarray_is_not_none(targets_noise):
            assert dictarray_is_not_none(features_noise)
            params['data']['num_targets'] += targets_noise['train'].shape[1]

    # print sample sizes
    logger.info(f"Ntrain:    {data_params['Ntrain']}")
    logger.info(f"Nvalidate: {data_params['Nvalidate']}")
    logger.info(f"Ntest:     {data_params['Ntest']}")

    # print data shapes
    logger.debug(f"num_features: {params['data']['num_features']}")
    logger.debug(f"num_targets:  {params['data']['num_targets']}")

    # return data
    return features, targets, features_noise, targets_noise

def load_timesteps(params):
    data_params = params['data']
    data_dir    = pathlib.Path(data_params['data_dir'])

    file_names = data_params.get('file_names', {})
    timesteps_name = file_names.get('timesteps')

    # allow explicit timesteps path for tar datasets
    if timesteps_name is not None:
        rel = _safe_format(str(timesteps_name), data_params)
        if data_dir.is_file():
            with tarfile.open(data_dir, mode="r:*") as tf:
                resolved = _resolve_tar_member_name(tf, rel, data_prefix=data_params.get('data_prefix'))
                if resolved is None:
                    raise FileNotFoundError(f"Tar member not found for timesteps: {rel} in {data_dir}")
                fh = tf.extractfile(resolved)
                if fh is None:
                    raise FileNotFoundError(f"Unable to open timesteps member: {resolved} in {data_dir}")
                return np.load(io.BytesIO(fh.read()))
        return np.load(data_dir / rel)

    # legacy fallbacks
    if '2020' in data_dir.name:
        return np.load(data_dir/'fhn_T200_samplePrior_time.npy')
    if '2025' in data_dir.name:
        return np.load(data_dir/'fhn_timesteps_Nt2000_dt0.2.npy')

    raise NotImplementedError(
        f"Unsupported data directory for timesteps: {data_dir}. "
        f"Provide data.file_names.timesteps for dataset_layout='{data_params.get('dataset_layout')}'."
    )

###############################################################################

#def _log_transform(data, shift=0.0):
#    """ Applies log-transform for preprocessing. """
#    if isinstance(data, dict):
#        for key in data.keys():
#            data[key] = np.log(shift + data[key])
#    else:
#        data = np.log(shift + data)
#    return data

#def _log_transform_inverse(data, shift=0.0):
#    """ Applies inverse of log-transform for postprocessing. """
#    if isinstance(data, dict):
#        for key in data.keys():
#            data[key] = np.exp(data[key]) - shift
#    else:
#        data = np.exp(data) - shift
#    return data

def _apply_scale(data, scale):
    """ Applies scale for preprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = (data[key] - scale['shift']) * (1.0/scale['mult'])
    else:
        data = (data - scale['shift']) * (1.0/scale['mult'])
    return data

def _apply_scale_inverse(data, scale):
    """ Applies inverse scale for postprocessing. """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = data[key] * scale['mult'] + scale['shift']
    else:
        data = data * scale['mult'] + scale['shift']
    return data


def _resolve_targets_transform_cfg(params, n_targets):
    """
    Resolve per-target transform configuration.

    Supported transform names:
      - identity
      - log10
      - log1p
      - signed_log1p

    Accepted config formats under data.targets_transform:
      - "log10"                    -> apply to all targets
      - ["log10", "identity", ...] -> explicit per-target list
      - {"names": [...]}           -> explicit per-target list
      - {"default": "identity", "per_target": {0: "log10", 3: "log10"}}
    """
    data_cfg = params.get("data", {}) if isinstance(params, dict) else {}
    raw = data_cfg.get("targets_transform")
    eps = float(data_cfg.get("targets_transform_eps", 1.0e-12))
    valid = {"identity", "log10", "log1p", "signed_log1p"}

    if raw is None:
        names = ["identity"] * int(n_targets)
    elif isinstance(raw, str):
        names = [raw.strip().casefold()] * int(n_targets)
    elif isinstance(raw, (list, tuple)):
        names = [str(x).strip().casefold() for x in raw]
        if len(names) != int(n_targets):
            raise ValueError(
                f"data.targets_transform list length mismatch: got {len(names)}, expected {n_targets}"
            )
    elif isinstance(raw, dict):
        if "names" in raw:
            names = [str(x).strip().casefold() for x in raw["names"]]
            if len(names) != int(n_targets):
                raise ValueError(
                    f"data.targets_transform.names length mismatch: got {len(names)}, expected {n_targets}"
                )
        else:
            default_name = str(raw.get("default", "identity")).strip().casefold()
            names = [default_name] * int(n_targets)
            per_target = raw.get("per_target", {}) or {}
            for k, v in per_target.items():
                idx = int(k)
                if idx < 0 or idx >= int(n_targets):
                    raise ValueError(
                        f"data.targets_transform.per_target index out of range: {idx} for n_targets={n_targets}"
                    )
                names[idx] = str(v).strip().casefold()
    else:
        raise ValueError(
            "data.targets_transform must be one of: string, list, tuple, dict, or null"
        )

    bad = [name for name in names if name not in valid]
    if bad:
        raise ValueError(
            f"Unsupported target transform(s): {bad}. Supported={sorted(valid)}"
        )

    return {"names": names, "eps": eps}


def _targets_transform_array(arr, transform_cfg, inverse=False, context="targets"):
    if arr is None or getattr(arr, "size", 0) == 0:
        return arr
    if arr.ndim < 2:
        raise ValueError(f"{context}: expected ndim>=2 for target transforms, got shape={arr.shape}")
    if not getattr(arr, "flags", None) or not arr.flags.writeable:
        # np.load(..., mmap_mode='r') yields read-only arrays; transforms below are in-place.
        arr = np.array(arr, copy=True)

    names = transform_cfg["names"]
    eps = float(transform_cfg.get("eps", 1.0e-12))
    if arr.shape[1] != len(names):
        raise ValueError(
            f"{context}: targets dimension mismatch for transforms: arr.shape={arr.shape}, "
            f"len(transform_names)={len(names)}"
        )

    for i, name in enumerate(names):
        if name == "identity":
            continue

        x = arr[:, i, ...]
        if not inverse:
            if name == "log10":
                if np.any(x <= eps):
                    bad = int(np.count_nonzero(x <= eps))
                    raise ValueError(
                        f"{context}: transform 'log10' on target[{i}] requires values>{eps}; "
                        f"found {bad} non-positive/near-zero values"
                    )
                arr[:, i, ...] = np.log10(x)
            elif name == "log1p":
                if np.any(x <= (-1.0 + eps)):
                    bad = int(np.count_nonzero(x <= (-1.0 + eps)))
                    raise ValueError(
                        f"{context}: transform 'log1p' on target[{i}] requires values>{-1.0 + eps}; "
                        f"found {bad} invalid values"
                    )
                arr[:, i, ...] = np.log1p(x)
            elif name == "signed_log1p":
                arr[:, i, ...] = np.sign(x) * np.log1p(np.abs(x))
            else:
                raise ValueError(f"{context}: unsupported transform '{name}'")
        else:
            if name == "log10":
                # Guard inverse-log transform against overflow/non-finite model outputs.
                # Example failure mode: very large x leads to inf after 10**x, then
                # sklearn metrics crash on non-finite predictions.
                arr_dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float32
                max_log10 = float(np.floor(np.log10(np.finfo(arr_dtype).max)))
                min_log10 = -max_log10

                x64 = np.asarray(x, dtype=np.float64)
                bad_nonfinite = int(np.count_nonzero(~np.isfinite(x64)))
                bad_hi = int(np.count_nonzero(np.isfinite(x64) & (x64 > max_log10)))
                bad_lo = int(np.count_nonzero(np.isfinite(x64) & (x64 < min_log10)))
                if (bad_nonfinite + bad_hi + bad_lo) > 0:
                    warnings.warn(
                        f"{context}: inverse 'log10' clipping target[{i}] to "
                        f"[{min_log10:.1f}, {max_log10:.1f}] "
                        f"(nonfinite={bad_nonfinite}, >max={bad_hi}, <min={bad_lo})",
                        RuntimeWarning,
                    )

                x_safe = np.nan_to_num(
                    x64,
                    nan=max_log10,
                    posinf=max_log10,
                    neginf=min_log10,
                )
                x_safe = np.clip(x_safe, min_log10, max_log10)
                arr[:, i, ...] = np.power(10.0, x_safe).astype(arr_dtype, copy=False)
            elif name == "log1p":
                arr[:, i, ...] = np.expm1(x)
            elif name == "signed_log1p":
                arr[:, i, ...] = np.sign(x) * np.expm1(np.abs(x))
            else:
                raise ValueError(f"{context}: unsupported inverse transform '{name}'")

    return arr


def _apply_targets_transform(data, transform_cfg, inverse=False, array_name="targets"):
    if transform_cfg is None:
        return data

    if isinstance(data, dict):
        for key in data.keys():
            if data[key] is None:
                continue
            data[key] = _targets_transform_array(
                data[key],
                transform_cfg,
                inverse=inverse,
                context=f"{array_name}['{key}']",
            )
    else:
        data = _targets_transform_array(
            data,
            transform_cfg,
            inverse=inverse,
            context=array_name,
        )
    return data

def preprocess_features(features, params, logger, scale=None, array_name='features'):
    # exit if nothing to do
    if dictarray_is_none(features):
        return None
    lazy_split_views = dictarray_uses_indexed_arrays(features)
    features_type = params['data']['features_type'].casefold()
    features_train = np.asarray(features['train']) if lazy_split_views else features['train']
    data_params = params['data']
    cache_path = None
# DEV
#   # apply transformation
#   if features_type == 'RATE_DURATION'.casefold():
#       for key in features.keys():
#           features[key][...,1] = _log_transform(features[key][...,1], shift=1.0)
#/DEV
    # calculate scaling for normalization
    if scale is None:
        cache_path = _resolve_features_scale_cache_path(data_params, array_name=array_name)
        dist_init, rank = _dist_rank_info()
        if cache_path is not None and cache_path.exists():
            try:
                scale = _load_features_scale_cache(cache_path)
                logger.info(f"{array_name} scale cache hit = {cache_path}")
            except Exception as exc:
                logger.warning(f"{array_name} scale cache load failed at {cache_path}: {exc}; recomputing")
                scale = None
        elif cache_path is not None and dist_init and rank != 0:
            try:
                import torch.distributed as dist
                dist.barrier()
                if cache_path.exists():
                    scale = _load_features_scale_cache(cache_path)
                    logger.info(f"{array_name} scale cache hit after rank0 write = {cache_path}")
            except Exception as exc:
                logger.warning(f"{array_name} scale cache synchronization failed at {cache_path}: {exc}; recomputing")
                scale = None
        shape = features_train.ndim * [1]
        shape[1] = features_train.shape[1]
        dtype = features_train.dtype
        if scale is None:
            scale = {
                'shift': np.zeros(shape, dtype=dtype),
                'mult':  np.ones(shape, dtype=dtype),
            }
            if data_params.get('features_normalize', False):
                if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
                    assert 3 == features_train.ndim
                    scale = {
                        'shift': np.mean(features_train, axis=(0,2), keepdims=True),
                        'mult' : np.std (features_train, axis=(0,2), keepdims=True)
                    }
                elif features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
                    assert 3 == features_train.ndim
                    assert 1 == features_train.shape[1]
                    scale = {
                        'shift': np.nanmean(features_train, axis=0, keepdims=True),
                        'mult' : np.nanstd (features_train, axis=0, keepdims=True)
                    }
# DEV
#               # override scaling of "spike rate"
#               if 'RATE_DURATION'.casefold() == features_type:
#                   features_min = np.nanmin(features_train, axis=0, keepdims=True)
#                   features_max = np.nanmax(features_train, axis=0, keepdims=True)
#                   scale['shift'][...,0] = features_min[...,0]
#                   scale['mult'][...,0]  = features_max[...,0] - features_min[...,0]
#/DEV
                else:
                    raise NotImplementedError(f"Unknown {features_type=}")

            if cache_path is not None:
                try:
                    if (not dist_init) or (rank == 0):
                        _save_features_scale_cache(cache_path, scale)
                        logger.info(f"{array_name} scale cache write = {cache_path}")
                    if dist_init and rank == 0:
                        import torch.distributed as dist
                        dist.barrier()
                except Exception as exc:
                    logger.warning(f"{array_name} scale cache write failed at {cache_path}: {exc}")
    logger.info(f"{array_name} scale = {scale}")
    if 3 < features_train.ndim:
        logger.warning(
            f"Scaling of features is currently only tested for ndim=3,"
            + f" got ndim={features_train.ndim}"
        )
    if lazy_split_views:
        logger.info(
            "%s uses lazy_split_views; deferring feature scaling/materialization to the dataloader",
            array_name,
        )
        return scale
    # apply scaling
    features = _apply_scale(features, scale)
    # replace nan values
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
        pass
    elif features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
        for key in features.keys():
            features[key] = np.where(np.isnan(features[key]), -10.0, features[key])
    else:
        raise NotImplementedError(f"Unknown {features_type=}")
    # return scale
    return scale

def postprocess_features(features, scale, params):
    # exit if nothing to do
    if dictarray_is_none(features):
        return
    features_type = params['data']['features_type'].casefold()
    # apply inverse scaling
    features = _apply_scale_inverse(features, scale)
# DEV
#   # apply inverse scaling
#   if features_type == 'RATE_DURATION'.casefold():
#       features = _apply_scale_inverse(features, scale)
#       # apply inverse transforms here if a downstream caller requires them
#       raise NotImplementedError()
#/DEV

def preprocess_targets(targets, params, logger, scale=None, array_name='targets'):
    # exit if nothing to do
    if dictarray_is_none(targets):
        return None
    lazy_split_views = dictarray_uses_indexed_arrays(targets)
    targets_train = np.asarray(targets['train']) if lazy_split_views else targets['train']
    n_targets = int(targets_train.shape[1])

    # resolve transform config (prefer explicit config from provided scale)
    if (scale is not None) and isinstance(scale, dict) and ("transform" in scale):
        transform_cfg = scale["transform"]
    else:
        transform_cfg = _resolve_targets_transform_cfg(params, n_targets)

    # apply transform before normalization/scaling
    if lazy_split_views:
        targets_train_transformed = _targets_transform_array(
            np.array(targets_train, copy=True),
            transform_cfg,
            inverse=False,
            context=f"{array_name}['train']",
        )
    else:
        targets = _apply_targets_transform(targets, transform_cfg, inverse=False, array_name=array_name)
        targets_train_transformed = targets['train']

    # calculate scaling for normalization
    if scale is None:
        shape = (1, *targets_train_transformed.shape[1:])
        dtype = targets_train_transformed.dtype
        scale = {
            'shift': np.zeros(shape, dtype=dtype),
            'mult':  np.ones(shape, dtype=dtype),
            'transform': transform_cfg,
        }
        if params['data'].get('targets_normalize', False):
            assert 1 < targets_train_transformed.ndim
            scale = {
                'shift': np.mean(targets_train_transformed, axis=0, keepdims=True),
                'mult' : np.std (targets_train_transformed, axis=0, keepdims=True),
                'transform': transform_cfg,
            }
    elif isinstance(scale, dict) and ("transform" not in scale):
        scale["transform"] = transform_cfg

    # protect against division by zero in degenerate target columns
    if isinstance(scale, dict) and ("mult" in scale):
        mult = np.asarray(scale["mult"])
        mult = np.where(mult == 0, 1.0, mult)
        scale["mult"] = mult

    logger.info(f"{array_name} target transform = {transform_cfg}")
    logger.info(f"{array_name} scale = {scale}")
    if 2 < targets_train_transformed.ndim:
        logger.warning(
            f"Scaling of targets is currently only tested for ndim=2,"
            + f" got ndim={targets_train_transformed.ndim}"
        )
    if lazy_split_views:
        logger.info(
            "%s uses lazy_split_views; deferring target scaling/materialization to the dataloader",
            array_name,
        )
        return scale
    # apply scaling
    targets = _apply_scale(targets, scale)
    # output scale
    return scale

def make_features_preprocess_transform(scale, params):
    if scale is None:
        return None

    features_type = params['data']['features_type'].casefold()
    shift = torch.as_tensor(scale['shift'], dtype=torch.float32)
    mult = torch.as_tensor(scale['mult'], dtype=torch.float32)

    def _transform(features_batch):
        shift_local = shift.to(device=features_batch.device, dtype=features_batch.dtype)
        mult_local = mult.to(device=features_batch.device, dtype=features_batch.dtype)
        out = (features_batch - shift_local) * (1.0 / mult_local)
        if features_type in ['ODE_STATS'.casefold(), 'RATE_DURATION'.casefold()]:
            out = torch.nan_to_num(out, nan=-10.0)
        return out

    return _transform


def _targets_transform_tensor(batch, transform_cfg):
    if transform_cfg is None:
        return batch

    names = transform_cfg["names"]
    eps = float(transform_cfg.get("eps", 1.0e-12))
    if batch.ndim < 2:
        raise ValueError(f"targets tensor transform expects ndim>=2, got shape={tuple(batch.shape)}")
    if batch.shape[1] != len(names):
        raise ValueError(
            f"targets tensor dimension mismatch for transforms: batch.shape={tuple(batch.shape)} len(names)={len(names)}"
        )

    out = batch.clone()
    for i, name in enumerate(names):
        if name == "identity":
            continue
        x = out[:, i, ...]
        if name == "log10":
            if torch.any(x <= eps):
                bad = int(torch.count_nonzero(x <= eps).item())
                raise ValueError(
                    f"targets tensor transform 'log10' on target[{i}] requires values>{eps}; found {bad} invalid values"
                )
            out[:, i, ...] = torch.log10(x)
        elif name == "log1p":
            if torch.any(x <= (-1.0 + eps)):
                bad = int(torch.count_nonzero(x <= (-1.0 + eps)).item())
                raise ValueError(
                    f"targets tensor transform 'log1p' on target[{i}] requires values>{-1.0 + eps}; found {bad} invalid values"
                )
            out[:, i, ...] = torch.log1p(x)
        elif name == "signed_log1p":
            out[:, i, ...] = torch.sign(x) * torch.log1p(torch.abs(x))
        else:
            raise ValueError(f"Unsupported tensor target transform '{name}'")
    return out


def make_targets_preprocess_transform(scale):
    if scale is None:
        return None

    shift = torch.as_tensor(scale['shift'], dtype=torch.float32)
    mult = torch.as_tensor(scale['mult'], dtype=torch.float32)
    transform_cfg = scale.get("transform") if isinstance(scale, dict) else None

    def _transform(targets_batch):
        out = _targets_transform_tensor(targets_batch, transform_cfg)
        shift_local = shift.to(device=out.device, dtype=out.dtype)
        mult_local = mult.to(device=out.device, dtype=out.dtype)
        return (out - shift_local) * (1.0 / mult_local)

    return _transform

def postprocess_targets(targets, scale):
    # exit if nothing to do
    if dictarray_is_none(targets):
        return
    # apply inverse scaling
    targets = _apply_scale_inverse(targets, scale)
    # apply inverse transform back to original units
    transform_cfg = scale.get("transform") if isinstance(scale, dict) else None
    targets = _apply_targets_transform(targets, transform_cfg, inverse=True, array_name='targets')

###############################################################################

def _get_positions_from_histogram(data, range, n_bins, relevant_bins_threshold):
    hist, bin_edges = np.histogram(data.flatten(), range=range, bins=n_bins)
    relevant_bin_indices = hist > relevant_bins_threshold
    relevant_bin_edges   = (bin_edges[:-1])[relevant_bin_indices]
    n_relevant_bins      = np.sum(relevant_bin_indices)
    if 10 < n_relevant_bins:
        cond_positions = np.linspace(relevant_bin_edges[0], relevant_bin_edges[-1], 5)
    else:
        cond_positions = relevant_bin_edges
    return cond_positions

def get_conditional_positions(features: np.ndarray, params):
    """ Tuned for data set 2020-12-09. """
    data_dir      = params['data']['data_dir']
    features_type = params['data']['features_type'].casefold()
    # set function parameters
    fn_params = {
        'n_bins': {
            'TIME':     None,
            'RATE':     1000,
            'DURATION':   25,
        },
        'range': {
            'TIME':     None,
#           'RATE':     [ 0.55, 0.95],
#           'DURATION': [-0.50, 0.50],
            'RATE':     [-0.5, 1.0],
            'DURATION': [-0.5, 0.5],
        },
        'relevant_bins_threshold': {
            'TIME':     None,
            'RATE':     features.shape[0] * 0.06,
            'DURATION': features.shape[0] * 0.01,
        }
    }
    # extract conditional positions
    assert 1 < features.shape[0]
    cond_positions = list()
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
        # set indices of samples as positions for conditionals
        cond_positions.append(np.arange(features.shape[0] - 6, features.shape[0], dtype=np.int32))
    elif features_type == 'ODE_STATS'.casefold():
        raise NotImplementedError()
    elif features_type == 'RATE_DURATION'.casefold():
        if '2020' in data_dir:
#           cond_positions.append(np.array([ 0.60,  0.70,  0.80,  0.90]))
#           cond_positions.append(np.array([-0.36, -0.34, -0.26, -0.18, -0.17]))
            cond_positions.append(np.array([-0.4205, 0.1795, 0.7795]))
            cond_positions.append(np.array([-0.34, -0.3, -0.26, -0.22]))
        elif '2025' in data_dir:
            cond_positions.append(np.array([-0.419, 0.175, 0.769]))
            cond_positions.append(np.array([-0.5, -0.26, 0.22, 0.46]))
        else:  # otherwise find values from histogram
            for i, key in enumerate(['RATE', 'DURATION']):
                cond_positions.append(_get_positions_from_histogram(
                        features[...,i],
                        fn_params['range'][key],
                        fn_params['n_bins'][key],
                        fn_params['relevant_bins_threshold'][key]
                ))
    else:
        raise ValueError(f"Unknown features_type: {features_type}")
    return cond_positions

def _filter_samples(features, targets, position, threshold):
    # filter features
    if features.shape[-1] == len(position):
        # if as many positions as features: threshold positions across all features
        for i, (pos, thresh) in enumerate(zip(position, threshold)):
            features_ = features[...,i].flatten()
            idx_thresh = np.logical_and((pos - thresh) < features_, features_ < (pos + thresh))
            if 0 == i:
                indices = idx_thresh
            else:
                indices = np.logical_and(indices, idx_thresh)
    elif 1 == len(position) and features.shape[1:] == position[0].shape[1:]:
        # if positions are samples: threshold by the normed distance to samples
        sample = position[0]
        thresh = threshold[0]
        assert 1 == sample.shape[0]
        distances = np.linalg.norm(features - sample, axis=(1, 2))
        indices = np.where(distances < thresh)[0]
    features_filtered = features[indices]
    # apply filter to targets
    if 2 == targets.ndim:
        targets_filtered = targets[indices]
    elif 3 == targets.ndim:
        targets_filtered = targets[:,indices,...]
    else:
        raise NotImplementedError(f"targets.ndim={targets.ndim}")
    return features_filtered, targets_filtered

def get_conditional_samples(features: np.ndarray, targets: np.ndarray, position, params):
    features_type = params['data']['features_type'].casefold()
    # extract conditional samples
    assert 1 < features.shape[0]
    if features_type in ['TIME'.casefold(), 'TIME_NOISE'.casefold(), 'NOISE'.casefold()]:
        threshold = [0.01 * np.prod(features.shape[1:])]
        features_cond, targets_cond = _filter_samples(features, targets, position, threshold)
    elif features_type == 'ODE_STATS'.casefold():
        raise NotImplementedError()
    elif features_type == 'RATE_DURATION'.casefold():
        #threshold = [0.05, 0.15]
        threshold = [0.6, 0.6]
        features_cond, targets_cond = _filter_samples(features, targets, position, threshold)
    else:
        raise ValueError(f"Unknown features_type: {features_type}")
    return features_cond, targets_cond

###############################################################################

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

class FHN_Dataset(Dataset):
    def __init__(
            self,
            features,
            targets,
            features_noise=None,
            targets_noise=None,
            features_additive_noise_std=0.0,
            features_multiplicative_noise_std=0.0,
            features_baseline_drift_std=0.0,
            features_mask_fraction=0.0,
            features_transform_fn=None,
            targets_transform_fn=None,
            features_sub_length=None,
            features_sub_begin_random=False,
            features_sub_step=None,
            noise_idx_random=True,
            item_return_order='yx'
    ):
        super().__init__()
        assert features is not None or features_noise is not None
        assert targets is None or features is not None
        assert targets_noise is None or features_noise is not None
        # set arrays from arguments
        if isinstance(features, IndexedArray):
            self.features = features.base
            self.features_is_numpy = True
            self.features_indices = torch.as_tensor(features.indices, dtype=torch.long)
        elif features is not None:
            self.features = torch.from_numpy(features)
            self.features_is_numpy = False
            self.features_indices = None
        else:
            self.features = None
            self.features_is_numpy = False
            self.features_indices = None
        if isinstance(targets, IndexedArray):
            self.targets = targets.base
            self.targets_is_numpy = True
            self.targets_indices = torch.as_tensor(targets.indices, dtype=torch.long)
            assert self.targets_indices.numel() == (
                self.features_indices.numel() if self.features_indices is not None else self.features.size(0)
            )
        elif targets is not None:
            self.targets = torch.from_numpy(targets)
            self.targets_is_numpy = False
            self.targets_indices = None
            assert self.targets.size(0) == (
                self.features_indices.numel() if self.features_indices is not None else self.features.size(0)
            )
        else:
            self.targets = None
            self.targets_is_numpy = False
            self.targets_indices = None
        if isinstance(features_noise, IndexedArray):
            self.features_noise = features_noise.base
            self.features_noise_is_numpy = True
            self.features_noise_indices = torch.as_tensor(features_noise.indices, dtype=torch.long)
        elif features_noise is not None:
            self.features_noise = torch.from_numpy(features_noise)
            self.features_noise_is_numpy = False
            self.features_noise_indices = None
        else:
            self.features_noise = None
            self.features_noise_is_numpy = False
            self.features_noise_indices = None
        if isinstance(targets_noise, IndexedArray):
            self.targets_noise = targets_noise.base
            self.targets_noise_is_numpy = True
            self.targets_noise_indices = torch.as_tensor(targets_noise.indices, dtype=torch.long)
        elif targets_noise is not None:
            self.targets_noise = torch.from_numpy(targets_noise)
            self.targets_noise_is_numpy = False
            self.targets_noise_indices = None
        else:
            self.targets_noise = None
            self.targets_noise_is_numpy = False
            self.targets_noise_indices = None
        # set from arguments
        self.features_additive_noise_std = features_additive_noise_std
        self.features_multiplicative_noise_std = features_multiplicative_noise_std
        self.features_baseline_drift_std = features_baseline_drift_std
        self.features_mask_fraction = features_mask_fraction
        self.features_transform_fn       = features_transform_fn
        self.targets_transform_fn        = targets_transform_fn
        self.features_sub_length         = features_sub_length
        self.features_sub_begin_random   = features_sub_begin_random
        self.noise_idx_random            = noise_idx_random
        self.item_return_order           = item_return_order.casefold()
        self.features_sub_step           = features_sub_step


    def __len__(self):
        if self.features is not None:
            return int(self.features_indices.numel()) if self.features_indices is not None else self.features.size(0)
        else:
            return int(self.features_noise_indices.numel()) if self.features_noise_indices is not None else self.features_noise.size(0)

    def _row_index(self, idx, indices):
        if indices is None:
            return idx
        return int(indices[idx].item())

    def _source_length(self, source, source_is_numpy, indices):
        if indices is not None:
            return int(indices.numel())
        return int(source.shape[0]) if source_is_numpy else int(source.size(0))

    def _source_row(self, source, row_idx, source_is_numpy):
        if source_is_numpy:
            return torch.as_tensor(np.array(source[row_idx], copy=True))
        return source[row_idx]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get feature sample
        if self.features is not None:
            features_row = self._row_index(idx, self.features_indices)
            features = self._source_row(self.features, features_row, self.features_is_numpy)
            if self.features_noise is not None:
                if self.noise_idx_random:
                    noise_pool = self._source_length(self.features_noise, self.features_noise_is_numpy, self.features_noise_indices)
                    noise_idx = int(torch.randint(noise_pool, (1,))[0].item())
                else:
                    noise_idx = idx
                noise_row = self._row_index(noise_idx, self.features_noise_indices)
                features_transformed = features + self._source_row(self.features_noise, noise_row, self.features_noise_is_numpy)
            else:
                noise_idx = None
                features_transformed = features
        elif self.features_noise is not None:
            noise_row = self._row_index(idx, self.features_noise_indices)
            features = self._source_row(self.features_noise, noise_row, self.features_noise_is_numpy)
            features_transformed = features
        else:
            features = None
            features_transformed = None
        # apply additive i.i.d. noise (avoid in-place writes; required for memmap read-only arrays)
        if self.features_additive_noise_std:
            features_transformed = features_transformed + (
                self.features_additive_noise_std * torch.randn(features_transformed.size())
            )
        # apply multiplicative i.i.d. noise in normalized feature space
        if self.features_multiplicative_noise_std:
            features_transformed = features_transformed * (
                1.0 + self.features_multiplicative_noise_std * torch.randn(features_transformed.size())
            )
        # apply a simple linear baseline drift per sample/channel
        if self.features_baseline_drift_std:
            drift_axis = torch.linspace(-1.0, 1.0, features_transformed.size(-1), dtype=features_transformed.dtype)
            drift_shape = features_transformed.shape[:-1] + (1,)
            drift_coeff = self.features_baseline_drift_std * torch.randn(drift_shape, dtype=features_transformed.dtype)
            features_transformed = features_transformed + drift_coeff * drift_axis
        # mask one contiguous segment to emulate missing observations / partial observability
        if self.features_mask_fraction:
            seg_len = max(1, int(round(float(self.features_mask_fraction) * features_transformed.size(-1))))
            seg_len = min(seg_len, features_transformed.size(-1))
            start_max = max(1, features_transformed.size(-1) - seg_len + 1)
            start = int(torch.randint(start_max, (1,), dtype=torch.long)[0].item())
            features_transformed = features_transformed.clone()
            features_transformed[..., start:start + seg_len] = 0.0

        # truncate features array
        if self.features_sub_length and \
           self.features_sub_length < features.size(-1):
            if self.features_sub_begin_random:
                idx_begin = np.random.randint(features.size(-1) - self.features_sub_length)
            else:
                idx_begin = 0
            idx_end = idx_begin + self.features_sub_length
            features             = features            [...,idx_begin:idx_end]
            features_transformed = features_transformed[...,idx_begin:idx_end]
        # truncate features with step length
        if self.features_sub_step and 1 < self.features_sub_step:
            features             = features            [...,::self.features_sub_step]
            features_transformed = features_transformed[...,::self.features_sub_step]
        # transform features
        if self.features_transform_fn is not None:
            features_transformed = self.features_transform_fn(features_transformed[None,...])[0]
        # get target sample
        if self.targets is not None:
            targets_row = self._row_index(idx, self.targets_indices)
            targets = self._source_row(self.targets, targets_row, self.targets_is_numpy)
            if self.targets_noise is not None:
                assert noise_idx is not None
                noise_row = self._row_index(noise_idx, self.targets_noise_indices)
                targets_noise = self._source_row(self.targets_noise, noise_row, self.targets_noise_is_numpy)
                targets = torch.cat((targets, targets_noise), dim=0)
        elif self.targets_noise is not None:
            noise_row = self._row_index(idx, self.targets_noise_indices)
            targets = self._source_row(self.targets_noise, noise_row, self.targets_noise_is_numpy)
        else:
            targets = None
        if self.targets_transform_fn is not None and targets is not None:
            targets = self.targets_transform_fn(targets[None,...])[0]
        # return sample
        if 'xx'.casefold() == self.item_return_order:
            assert targets is not None
            return (targets, targets)
        elif 'xy'.casefold() == self.item_return_order:
            assert features_transformed is not None and targets is not None
            return (targets, features_transformed)
        elif 'yx'.casefold() == self.item_return_order:
            assert features_transformed is not None and targets is not None
            return (features_transformed, targets)
        elif 'yy'.casefold() == self.item_return_order:
            assert features_transformed is not None and features is not None
            return (features_transformed, features)
        else:
            raise ValueError(f"Unknown item return order: {self.item_return_order}")


def create_dataloader(params, logger, mode,
                      features, targets,
                      features_noise, targets_noise,
                      features_transform_fn=None,
                      targets_transform_fn=None,
                      item_return_order='yx',
                      distributed=False,
                      rank=0,
                      world_size=1):
    """ Creates a PyTorch dataset and dataloader from numpy arrays.
        Ref: https://pytorch.org/docs/stable/data.html
    """
    if mode.any(Mode.TRAIN | Mode.PROFILE):
        shuffle = True
        batch_size = params['data']['train_batch_size']
        is_train = True
    elif mode.any(Mode.VALIDATE | Mode.PREDICT | Mode.EVAL):
        shuffle = False
        batch_size = params['data']['eval_batch_size']
        is_train = False
    else:
        raise NotImplementedError()

    # set arguments for dataset
    dataset_kwargs = dict(noise_idx_random=shuffle)

    # create the dataset
    logger.info('Create new dataset')
    dataset = FHN_Dataset(
        features, targets,
        features_noise=features_noise,
        targets_noise=targets_noise,
        features_additive_noise_std=params['data'].get('features_additive_noise_std', 0.0),
        features_multiplicative_noise_std=params['data'].get('features_multiplicative_noise_std', 0.0),
        features_baseline_drift_std=params['data'].get('features_baseline_drift_std', 0.0),
        features_mask_fraction=params['data'].get('features_mask_fraction', 0.0),
        features_transform_fn=features_transform_fn,
        targets_transform_fn=targets_transform_fn,
        features_sub_length=params['data'].get('features_sub_length', 0),
        features_sub_begin_random=(
            params['data'].get('features_sub_begin_random', False)
            if is_train else
            params['data'].get('features_sub_begin_random_eval', False)
        ),
        features_sub_step=params['data'].get('features_sub_step'),
        item_return_order=item_return_order,
        **dataset_kwargs
    )

    # ----------------------------
    # Distributed sampler (training only)
    # ----------------------------
    sampler = None
    if distributed and (world_size > 1) and is_train:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False
        )

    # set arguments for dataloader
    dataloader_kwargs = dict(
        drop_last=False,
        batch_size=batch_size,
    )

    # if using a sampler, DataLoader shuffle must be False
    if sampler is not None:
        dataloader_kwargs.update(dict(
            sampler=sampler,
            shuffle=False,
        ))
    else:
        dataloader_kwargs.update(dict(
            shuffle=shuffle,
        ))

    # prefer Slurm allocation over full node capacity
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    avail_cpus_total = int(slurm_cpus) if slurm_cpus is not None else (os.cpu_count() or 1)

    # avoid oversubscribing: SLURM_CPUS_PER_TASK is per node, so split by LOCAL_WORLD_SIZE (per-node ranks),
    # not by WORLD_SIZE (global ranks), otherwise multi-node runs will underutilize CPU workers.
    local_world_size_env = os.environ.get("LOCAL_WORLD_SIZE")
    try:
        local_world_size = int(local_world_size_env) if local_world_size_env is not None else 1
    except ValueError:
        local_world_size = 1

    if distributed:
        denom = local_world_size if local_world_size > 0 else 1
    else:
        denom = 1

    avail_cpus = max(1, avail_cpus_total // denom)

    # deterministic-ish worker seeding per rank
    base_seed = params['data'].get('random_seed', None)

    def _seed_worker(worker_id: int):
        if base_seed is None:
            return
        seed = int(base_seed) + 1000 * int(rank) + int(worker_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    generator = None
    if base_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(base_seed) + 1000 * int(rank))

    if torch.cuda.is_available():
        default_n_workers = max(1, min(16, avail_cpus // 2))
        default_pin_memory = True
    else:
        default_n_workers = max(1, min(avail_cpus, 2))
        default_pin_memory = False

    worker_override = params['data'].get('dataloader_num_workers', None)
    n_workers = int(worker_override) if worker_override is not None else default_n_workers
    if n_workers < 0:
        raise ValueError(f"data.dataloader_num_workers must be >= 0, got {n_workers}")

    pin_memory = bool(params['data'].get('dataloader_pin_memory', default_pin_memory))
    persistent_workers = bool(params['data'].get('dataloader_persistent_workers', True))
    multiprocessing_context = params['data'].get('dataloader_multiprocessing_context', 'fork')
    prefetch_factor = params['data'].get('dataloader_prefetch_factor', 2)

    dataloader_kwargs.update(dict(
        num_workers=n_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    ))

    if n_workers > 0:
        dataloader_kwargs.update(dict(
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        ))
        if prefetch_factor is not None:
            dataloader_kwargs['prefetch_factor'] = int(prefetch_factor)

    # create the dataloader
    logger.info(
        'Create new dataloader (num_workers=%s pin_memory=%s persistent_workers=%s multiprocessing_context=%s prefetch_factor=%s)',
        n_workers,
        pin_memory,
        (persistent_workers if n_workers > 0 else False),
        (multiprocessing_context if n_workers > 0 else None),
        (prefetch_factor if n_workers > 0 else None),
    )
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    return dataloader
