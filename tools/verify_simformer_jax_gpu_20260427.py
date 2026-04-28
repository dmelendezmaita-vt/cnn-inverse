#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify that the dedicated Simformer JAX environment can execute on GPU.")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    size = int(args.size)
    repeats = int(args.repeats)
    x = jnp.arange(size * size, dtype=jnp.float32).reshape(size, size)
    y = None
    for _ in range(repeats):
        y = jnp.dot(x, x.T).block_until_ready()
    assert y is not None
    print(
        json.dumps(
            {
                "jax_default_backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "device_count": jax.device_count(),
                "size": size,
                "repeats": repeats,
                "sum": float(jnp.sum(y)),
            }
        )
    )


if __name__ == "__main__":
    main()
