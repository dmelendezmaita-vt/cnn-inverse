# HH Track4 Simformer Branch

- official upstream Simformer codepath on the upstream HH task, reduced local budget
- not directly comparable to the local Track4 fixed-array workflow

| run | num_simulations | max_number_steps | training_batch_size | posterior_steps | num_eval_observations | num_samples | mae | mse | r2 | train_runtime_sec | sample_runtime_sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `primary` | 512 | 120 | 32 | 50 | 5 | 128 | 9.160283 | 159.608505 | 0.953334 | 28.595793 | 5.611550 |
| `secondary` | 512 | 120 | 32 | 50 | 5 | 128 | 7.440014 | 93.181946 | 0.967805 | 28.916363 | 5.619275 |
