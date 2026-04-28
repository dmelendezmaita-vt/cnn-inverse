# HH Track4 Generalized-Bayes NPE Branch

| field | value |
|---|---|
| `method_family` | `generalized_bayes_npe_local_beta_conditioned` |
| `adaptation_note` | `This is a bounded assumption-conditioned adaptation of beta-conditioned generalized-Bayes NPE, using SNIS-style tempered Gibbs weights over a prior simulation bank for each observed surrogate bundle.` |
| `n_base` | `1021` |
| `n_train_obs` | `8` |
| `n_test_obs` | `4` |
| `epochs` | `60` |
| `batch_size` | `128` |
| `learning_rate` | `0.001` |
| `hidden_dim` | `256` |
| `num_hidden_layers` | `2` |
| `runtime_sec` | `78.62147188186646` |
| `training_loss_final` | `0.0045561616577742825` |

| beta | mean_abs_log10_error_params_mean | abs_log10_error_current_gain_mean | posterior_std_norm_mean_mean |
|---:|---:|---:|---:|
| 0.25 | 35.248219 | 14.484261 | 20.085539 |
| 0.50 | 35.246421 | 14.485685 | 20.085539 |
| 1.00 | 35.242844 | 14.488529 | 20.085539 |
| 2.00 | 35.235664 | 14.494205 | 20.085539 |
