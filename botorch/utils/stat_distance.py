import torch
from torch import Tensor


def wasserstein_distance(
    p_mean: Tensor, q_mean: Tensor, p_var: Tensor, q_var: Tensor
) -> Tensor:
    """Computes

    Args:
        p_mean (Tensor): [batch_shape] x sample_shape tensor of means of first dist
        q_mean (Tensor): [batch_shape] x sample_shape tensor of variances of first dist
        p_var (Tensor): [batch_shape] x sample_shape tensor of means of second dist
        q_var (Tensor): [batch_shape] x sample_shape tensor of variances of second dist

    Returns:
        Tensor: The wasserstein distance between the gaussian distributions p and q
    """
    mean_term = torch.pow(p_mean - q_mean, 2)
    # rounding errors (rarely) occur, where the var term is ~-1e-16
    var_term = (p_var + q_var - 2 * torch.sqrt(p_var * q_var)).clamp_min(0)
    return torch.sqrt(mean_term + var_term)


def kl_divergence(
    p_mean: Tensor, q_mean: Tensor, p_var: Tensor, q_var: Tensor
) -> Tensor:
    """Computes

    Args:
        p_mean (Tensor): [batch_shape] x sample_shape tensor of means of first dist
        q_mean (Tensor): [batch_shape] x sample_shape tensor of variances of first dist
        p_var (Tensor): [batch_shape] x sample_shape tensor of means of second dist
        q_var (Tensor): [batch_shape] x sample_shape tensor of variances of second dist

    Returns:
        Tensor: The kl divergence between the gaussian distributions p and q
    """
    kl_first_term = torch.log(torch.sqrt(p_var / q_var))
    kl_second_term = 0.5 * (torch.pow(p_mean - q_mean, 2) + q_var) / p_var
    return kl_first_term + kl_second_term - 0.5


def hellinger_distance(
    p_mean: Tensor, q_mean: Tensor, p_var: Tensor, q_var: Tensor
) -> Tensor:
    """Computes

    Args:
        p_mean (Tensor): [batch_shape] x sample_shape tensor of means of first dist
        q_mean (Tensor): [batch_shape] x sample_shape tensor of variances of first dist
        p_var (Tensor): [batch_shape] x sample_shape tensor of means of second dist
        q_var (Tensor): [batch_shape] x sample_shape tensor of variances of second dist

    Returns:
        Tensor: The hellinger distance between the gaussian distributions p and q
    """
    exp_term = -0.25 * torch.pow(p_mean - q_mean, 2) / (p_var + q_var)
    mult_term = torch.sqrt(2 * torch.sqrt(p_var * q_var) / (p_var + q_var))
    return torch.sqrt(1 - mult_term * torch.exp(exp_term))
