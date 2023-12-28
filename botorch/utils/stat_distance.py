import torch
from torch import Tensor


def wasserstein_distance(
    p_mean: Tensor, q_mean: Tensor, p_covar: Tensor, q_covar: Tensor
) -> Tensor:
    """Computes

    Args:
        p_mean (Tensor): [batch_shape] x dist_shape tensor of means of first dist
        q_mean (Tensor): [batch_shape] x dist_shape tensor of variances of first dist
        p_var (Tensor): [batch_shape] x dist_shape tensor of means of second dist
        q_var (Tensor): [batch_shape] x dist_shape tensor of variances of second dist

    Returns:
        Tensor: The wasserstein distance between the gaussian distributions p and q
    """
    mean_term = torch.pow(p_mean - q_mean, 2)
    # rounding errors (rarely) occur, where the var term is ~-1e-16
    var_term = (p_var + q_var - 2 * torch.sqrt(p_var * q_var)).clamp_min(0)
    return torch.sqrt(mean_term + var_term)


def kl_divergence(
    p_mean: Tensor, q_mean: Tensor, p_covar: Tensor, q_covar: Tensor
) -> Tensor:
    """Computes

    Args:
        p_mean (Tensor): [batch_shape] x dist_shape tensor of means of first dist
        q_mean (Tensor): [batch_shape] x dist_shape tensor of variances of first dist
        p_var (Tensor): [batch_shape] x dist_shape tensor of means of second dist
        q_var (Tensor): [batch_shape] x dist_shape tensor of variances of second dist

    Returns:
        Tensor: The kl divergence between the gaussian distributions p and q
    """
    breakpoint()
    kl_first_term = torch.log(torch.sqrt(p_var / q_var))
    kl_second_term = 0.5 * (torch.pow(p_mean - q_mean, 2) + q_var) / p_var
    return kl_first_term + kl_second_term - 0.5


def hellinger_distance(
    p_mean: Tensor, q_mean: Tensor, p_covar: Tensor, q_covar: Tensor
) -> Tensor:
    """Computes

    Args:
        p_mean (Tensor): [batch_shape] x dist_shape tensor of means of first dist
        q_mean (Tensor): [batch_shape] x dist_shape tensor of variances of first dist
        p_var (Tensor): [batch_shape] x dist_shape tensor of means of second dist
        q_var (Tensor): [batch_shape] x dist_shape tensor of variances of second dist

    Returns:
        Tensor: The hellinger distance between the gaussian distributions p and q
    """
    p_logdet = torch.logdet(p_covar).unsqueeze(-1)
    q_logdet = torch.logdet(q_covar).unsqueeze(-1)
    avg_covar = (p_covar + q_covar) / 2

    # we need to re-use the cholesky decomp, so we compute it once here 
    L_avg = torch.linalg.cholesky(avg_covar)
    L_inv = torch.inverse(L_avg)

    # removes one dimension, which needs to be recouped
    pq_logdet = torch.pow(torch.diagonal(L_avg, dim1=-2, dim2=-1), 2).prod(
        dim=-1, keepdim=True).log()
    base_logterm = 0.25 * (p_logdet + q_logdet) - 0.5 * pq_logdet
    
    mean_diff = p_mean - q_mean

    L_mean_diff = torch.matmul(L_inv, mean_diff)
    exp_logterm = -0.125 * torch.matmul(L_mean_diff.transpose(-2, -1), L_mean_diff)
    sq_hdist = 1 - (base_logterm + exp_logterm.squeeze(-1)).exp()
    return sq_hdist.sqrt()


def hellinger_distance_single(
    p_mean: Tensor, q_mean: Tensor, p_var: Tensor, q_var: Tensor
) -> Tensor:
    """Computes

    Args:
        p_mean (Tensor): [batch_shape] x dist_shape tensor of means of first dist
        q_mean (Tensor): [batch_shape] x dist_shape tensor of variances of first dist
        p_var (Tensor): [batch_shape] x dist_shape tensor of means of second dist
        q_var (Tensor): [batch_shape] x dist_shape tensor of variances of second dist

    Returns:
        Tensor: The hellinger distance between the gaussian distributions p and q
    """
    exp_term = -0.25 * torch.pow(p_mean - q_mean, 2) / (p_var + q_var)
    mult_term = torch.sqrt(2 * torch.sqrt(p_var * q_var) / (p_var + q_var))
    return torch.sqrt(1 - mult_term * torch.exp(exp_term))


