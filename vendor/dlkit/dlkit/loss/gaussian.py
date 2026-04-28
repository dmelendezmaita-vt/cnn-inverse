import torch

def _construct_covariance_from_diag_std(std_diag):
    return torch.diag_embed(std_diag*std_diag)

def gaussian_loss_fn(outputs, targets, svd_ranks=8, inv_eps=1.0e-5):
    y_mean, y_std_diag = outputs
    y_data = torch.flatten(targets, 1)  # Flatten from dim 1 : -1
    # get covariance matrix
    Cov = _construct_covariance_from_diag_std(y_std_diag)
    # approximate SVD
    U, S, V = torch.svd_lowrank(Cov, q=svd_ranks, niter=2)
    # calculate covariance-weighted inner-product
    y_diff = y_data - y_mean
    yt_Cinv_y = torch.unsqueeze(y_diff, 1) @ \
                V @ torch.diag_embed(1.0/(S + inv_eps)) @ torch.transpose(U, 1, 2) @ \
                torch.unsqueeze(y_diff, 2)
    # calculate Gaussian loss
    exp_part = yt_Cinv_y.squeeze_([1, 2])
    log_part = torch.sum(torch.log(S), 1)
    return torch.mean(exp_part + log_part)


