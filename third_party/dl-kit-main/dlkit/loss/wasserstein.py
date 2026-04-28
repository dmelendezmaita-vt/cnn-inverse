import torch

def wasserstein_loss_fn(d_outputs_gen, d_outputs_data):
    """ Calculates loss corresponding to a Wasserstein distance. """
    loss_gen  = torch.mean(d_outputs_gen)  if d_outputs_gen  is not None else 0.0
    loss_data = torch.mean(d_outputs_data) if d_outputs_data is not None else 0.0
    w_loss = loss_data - loss_gen   # value to be maximized
    return -w_loss, loss_gen        # value to be minimized

def wasserstein_reg_fn(d_net, x_gen, x_data, y_data, p=2, c0=1.0, eps=0.0, device=None, dlog=None):
    """ Computes regularization for network representing Wasserstein distance.
    This is a penalty term for Lipschitz continuity.

    Author: Deep Ray
    """
    batch_size, *other_dims = x_data.size()
    epsilon = torch.rand([batch_size] + [1]*len(other_dims), device=device)
    epsilon = epsilon.expand(-1, *other_dims)
    x_hat = epsilon * x_data + (1.0 - epsilon) * x_gen
    x_hat.requires_grad = True
    y_data.requires_grad = True
    d_outputs_hat = d_net(x_hat, y_data)
    # compute gradient
    grad = torch.autograd.grad(
        outputs=d_outputs_hat,
        inputs=(x_hat, y_data),
        grad_outputs=torch.ones_like(d_outputs_hat, device=device),
        create_graph=True,
        retain_graph=True
    )
    grad_x, grad_y = grad[0], grad[1]
    grad_x = grad_x.view(batch_size, -1)
    grad_y = grad_y.view(batch_size, -1)
    # compute the squared l2-norm of the gradient
    grad_norm_sq = eps + torch.add(torch.sum(torch.square(grad_x), dim=1),
                                   torch.sum(torch.square(grad_y), dim=1))
    # <code id="gradient_penalty_v0">
    #grad_norm    = torch.sqrt(grad_norm_sq)
    #grad_penalty = torch.pow(grad_norm - c0, p).mean()
    # </code>
    # <code id="gradient_penalty_v1">
    #grad_norm    = torch.sqrt(grad_norm_sq)
    #grad_penalty = torch.pow(torch.nn.functional.relu(grad_norm - c0), p).mean()
    # </code>
    # <code id="gradient_penalty_v2">
    grad_norm    = torch.sqrt(grad_norm_sq.detach())  # for logging purposes
    grad_penalty = torch.nn.functional.relu(grad_norm_sq - c0*c0).mean()
    # </code>
    # log to dictionary
    if dlog is not None:
        assert isinstance(dlog, dict), type(dlog)
        dlog['grad_norm'] = grad_norm.detach().mean().item()
    # return the value of the regularization term
    return grad_penalty

