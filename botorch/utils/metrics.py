import torch
from torch import Tensor


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
    p_inv_covar = torch.inverse(p_covar)
    mean_diff = p_mean - q_mean
    kl_first_term = torch.diagonal(torch.matmul(p_inv_covar, q_covar),
                                dim1=2, dim2=-1).sum(-1, keepdim=True)
    kl_second_term = torch.matmul(torch.matmul(mean_diff.transpose(-2, -1),
        p_inv_covar), mean_diff).squeeze(-1)
    kl_third_term = (torch.logdet(p_covar) - torch.logdet(q_covar)).unsqueeze(-1)
    return 0.5 * (kl_first_term + kl_second_term + kl_third_term - + p_mean.shape[-2]) 


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