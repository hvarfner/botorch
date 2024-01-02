r"""
Acquisition function for Self-Correcting Bayesian Optimization (SCoreBO).

.. [Hvarfner2023scorebo]
    C. Hvarfner, F. Hutter, L. Nardi,
    Self-Correcting Bayesian Optimization thorugh Bayesian Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2023.

"""

from __future__ import annotations

import warnings

from typing import Optional

import torch
from botorch import settings
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.acquisition.bayesian_active_learning import (
    DISTANCE_METRICS,
    FullyBayesianAcquisitionFunction,
)

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.fully_bayesian import MCMC_DIM, SaasFullyBayesianSingleTaskGP

from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL

from botorch.models.utils import fantasize as fantasize_flag
from botorch.sampling import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform, concatenate_pending_points
from torch import Tensor

# The lower bound on the CDF value of the max-values
CLAMP_LB = 1e-6


class qSelfCorrectingBayesianOptimization(
    FullyBayesianAcquisitionFunction, MCSamplerMixin
):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        optimal_outputs: Tensor,
        optimal_inputs: Optional[Tensor] = None,
        X_pending: Optional[Tensor] = None,
        distance_metric: str = "hellinger",
        estimation_type: str = "LB",
        sampler: Optional[MCSampler] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        num_samples: int = 64,
        maximize: bool = True,
    ) -> None:
        r"""Joint entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            optimal_inputs: A `num_samples x num_models x d`-dim tensor containing
                the sampled optimal inputs of dimension `d`.
            optimal_outputs: A `num_samples x num_models x 1`-dim Tensor containing
                the optimal objective values.
            X_pending (Optional[Tensor], optional): _description_. Defaults to None.
            distance_metric (str, optional): The distance metric used to calculate
            disagreement in the posterior. Defaults to "hellinger" (recommended).
            estimation_type: estimation_type: A string to determine which entropy
                estimate is computed: Lower bound" ("LB") or "Monte Carlo" ("MC").
                Lower Bound is recommended due to the comparable empirical performance.
            maximize: If true, we consider a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            num_samples: The number of Monte Carlo samples used for the Monte Carlo
                estimate.
        """

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
            self.conditional_model = self.model.condition_on_observations(
                X=self.model.transform_inputs(self.optimal_inputs),
                Y=self.optimal_outputs,
                noise=torch.full_like(self.optimal_outputs, MIN_INFERRED_NOISE_LEVEL),
            )

        # otherwise, we do a MES-like variant (which places vastly more emphasis on
        # HP learning as supposed to optimization)
        else:
            self.conditional_model = self.model

        # Currently only supports LB (lower bound) estimation, will add MC later on
        if estimation_type not in ["LB"]:
            raise ValueError(f"Estimation type {estimation_type} does not exist.")
        self.estimation_type = estimation_type

        # the default number of MC samples (512) are too many when doing FB modeling.
        if distance_metric not in DISTANCE_METRICS.keys():
            raise ValueError(
                f"Distance metric need to be one of {list(DISTANCE_METRICS.keys())}"
            )
        self.distance = DISTANCE_METRICS[distance_metric]
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # since we have two MC dims (over models and optima), we need to
        # unsqueeze a second dim to accomodate the posterior pass
        posterior = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=True
        )
        cond_means = posterior.mean
        marg_mean = cond_means.mean(dim=MCMC_DIM, keepdim=True)
        cond_variances = posterior.variance
        cond_covar = posterior.covariance_matrix
        # the mixture variance is squeezed, need it unsqueezed
        marg_covar = posterior.mixture_covariance_matrix.unsqueeze(MCMC_DIM)
        marg_var = posterior.mixture_variance.unsqueeze(MCMC_DIM)
        noiseless_var = self.conditional_model.posterior(
            X.unsqueeze(MCMC_DIM), observation_noise=False
        ).variance

        normalized_mvs = (self.optimal_outputs - cond_means) / noiseless_var.sqrt()

        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))

        mean_truncated = cond_means - noiseless_var.sqrt() * pdf_mvs / cdf_mvs

        # This is the noiseless variance (i.e. the part that gets truncated)
        var_truncated = noiseless_var * (
            1 - normalized_mvs * pdf_mvs / cdf_mvs - torch.pow(pdf_mvs / cdf_mvs, 2)
        )
        # and add the (possibly heteroskedastic) noise
        var_truncated = var_truncated + (cond_variances - noiseless_var)
        
        # truncating the entire covariance matrix is not trivial, so we assume the 
        # truncation is proportional on the off-diags as on the diagonals and scale
        # the covariance accordingly for all elements in the q-batch (if there is one)
        # for q=1, this is equivalent to simply truncating the posterior
        covar_scaling = (var_truncated / cond_variances).sqrt()
        trunc_covar = covar_scaling.transpose(-1, -2) * covar_scaling * cond_covar
        dist = self.distance(mean_truncated, marg_mean, trunc_covar, marg_covar)

        # squeeze output dim and average over optimal samples dim (MCMC_DIM).
        # Model dim is averaged later
        return dist.mean(MCMC_DIM).sum(-1)