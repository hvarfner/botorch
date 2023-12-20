r"""
Acquisition function for Self-Correcting Bayesian Optimization (SCoreBO).

.. [Hvarfner2023scorebo]
    C. Hvarfner, F. Hutter, L. Nardi,
    Self-Correcting Bayesian Optimization thorugh Bayesian Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2023.

"""

from __future__ import annotations

from typing import Optional
import warnings

import torch
from botorch import settings
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.acquisition.bayesian_active_learning import FullyBayesianAcquisitionFunction

from botorch.acquisition.objective import PosteriorTransform
from botorch.acquisition.bayesian_active_learning import DISTANCE_METRICS

from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP, MCMC_DIM

from botorch.models.utils import check_no_nans, fantasize as fantasize_flag
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor

from torch.distributions import Normal

# The lower bound on the CDF value of the max-values
CLAMP_LB = 1e-6

class SelfCorrectingBayesianOptimization(
        FullyBayesianAcquisitionFunction, MCSamplerMixin):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        optimal_outputs: Optional[Tensor],
        optimal_inputs: Optional[Tensor] = None,
        X_pending: Optional[Tensor] = None,
        distance_metric: str = "hellinger",
        estimation_type: str = "LB",
        sampler: Optional[MCSampler] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        num_samples: int = 64,
        maximize: bool = True
    ) -> None:
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        MCSamplerMixin.__init__(self, sampler=sampler)
        
        # To enable fully bayesian GP conditioning, we need to unsqueeze
        # to get num_optima x num_gps unique GPs
        self.posterior_transform = posterior_transform
        self.maximize = maximize
        if not self.maximize:
            optimal_outputs = -optimal_outputs

        # inputs come as num_optima_per_model x num_models x d
        # but we want it four-dimensional to condition one per model.
        
        self.optimal_outputs = optimal_outputs.unsqueeze(-2)
        # JES-like version of SCoreBO if optimal inputs are provided
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with fantasize_flag():
                with settings.propagate_grads(False):
                    # We must do a forward pass one before conditioning.
                    self.model.posterior(
                        self.model.train_inputs[0], observation_noise=False
                    )
        if optimal_inputs is not None:
            self.optimal_inputs = optimal_inputs.unsqueeze(-2)
            self.conditional_model = (
                self.model.condition_on_observations(
                    X=self.model.transform_inputs(self.optimal_inputs),
                    Y=self.optimal_outputs,
                    noise=torch.full_like(g
                        self.optimal_outputs, MIN_INFERRED_NOISE_LEVEL),
                )
            )
            
        # otherwise, we do a MES-like variant (which places vastly more emphasis on
        # HP learning as supposed to optimization)
        else:
            self.conditional_model = self.model
            
        if estimation_type not in ["LB", "MC"]:
            raise ValueError(f"Estimation type {estimation_type} does not exist.")
        self.estimation_type = estimation_type
        
        # the default number of MC samples (512) are too many when doing FB modeling.
        if distance_metric not in DISTANCE_METRICS.keys():
            raise ValueError(
                f"Distance metric need to be one of " f"{list(DISTANCE_METRICS.keys())}"
            )
        self.distance = DISTANCE_METRICS[distance_metric]
        self.set_X_pending(X_pending)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # since we have two MC dims (over models and optima), we need to
        # unsqueeze a second dim to accomodate the posterior pass
        posterior = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=True)
        breakpoint()
        cond_means = posterior.mean
        marg_mean = cond_means.mean(dim=MCMC_DIM, keepdim=True)
        cond_variances = posterior.variance
        # the mixture variance is squeezed, need it unsqueezed
        marg_variance = posterior.mixture_variance.unsqueeze(MCMC_DIM)
        noiseless_var = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=False).variance
        
        normalized_mvs = (self.optimal_outputs - cond_means) / noiseless_var.sqrt()
        
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))

        mean_truncated = cond_means - noiseless_var.sqrt() * pdf_mvs / cdf_mvs

        # This is the noiseless variance (i.e. the part that gets truncated)
        var_truncated = cond_variances * \
            (1 - normalized_mvs * pdf_mvs / cdf_mvs - torch.pow(pdf_mvs / cdf_mvs, 2))
        var_truncated = var_truncated + (cond_variances - cond_variances)


        dist = self.distance(cond_means, marg_mean, cond_variances, marg_variance)
        # squeeze output dim and average over optimal samples dim (MCMC_DIM).
        # Model dim is averaged later
        return dist.squeeze(-1).mean(MCMC_DIM).mean(-1)
