#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.models.fully_bayesian import MCMC_DIM, SaasFullyBayesianSingleTaskGP
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor


SAMPLE_DIM = -4


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


# TODO move to utils/distance_metrics
DISTANCE_METRICS = {
    "hellinger": hellinger_distance,
    "wasserstein": wasserstein_distance,
    "kl_divergence": kl_divergence,
}


class FullyBayesianAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: Model):
        """Base class for acquisition functions which require a Fully Bayesian
            model treatment.

        Args:
            AcquisitionFunction (_type_): _description_

        Returns:
            _type_: _description_

        Returns:
            _type_: _description_
        """
        if isinstance(model, SaasFullyBayesianSingleTaskGP):
            super().__init__(model)

        else:
            raise ValueError(
                "Fully Bayesian acquiition functions require "
                "a SaasFullyBayesianSingleTaskGP to run."
            )


class BayesianVarianceReduction(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        """_summary_

        Args:
            model (Model): _description_
        """
        super().__init__(model)

        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        return posterior.mixture_variance.squeeze(-1)


class BayesianQueryByComittee(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        """

        Args:
            model (Model): _description_
        """
        super().__init__(model)
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        posterior_mean = posterior.mean
        marg_mean = posterior_mean.mean(dim=MCMC_DIM, keepdim=True)
        var_of_mean = torch.pow(marg_mean - posterior_mean, 2)
        return var_of_mean.squeeze(-1).squeeze(-1)


class BayesianActiveLearningByDisagreement(
    FullyBayesianAcquisitionFunction, MCSamplerMixin
):
    def __init__(
        self,
        model: Model,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "MC",
        sampler: Optional[MCSampler] = None,
        num_samples: int = 64,
    ) -> None:
        """_summary_

        Args:
            model (Model): _description_
            X_pending (Optional[Tensor], optional): _description_. Defaults to None.
            estimation_type (str, optional): _description_. Defaults to "MC".
            sampler (Optional[MCSampler], optional): _description_. Defaults to None.
            num_samples (int, optional): _description_. Defaults to 64.

        Raises:
            ValueError: _description_
        """
        super().__init__(model)
        if estimation_type not in ["MC", "LB"]:
            raise ValueError(f"Estimation type {estimation_type} does not exist.")
        self.estimation_type = estimation_type
        self.set_X_pending(X_pending)
        # the default number of MC samples (512) are too many when doing FB modeling.
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        MCSamplerMixin.__init__(self, sampler=sampler)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        if self.estimation_type == "LB":
            return self._compute_lower_bound_information_gain(X)
        elif self.estimation_type == "MC":
            return self._compute_monte_carlo_information_gain(X)

    def _compute_lower_bound_information_gain(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        marg_variance = posterior.mixture_variance.unsqueeze(-1)
        cond_variances = posterior.variance
        bald = torch.log(marg_variance) - torch.log(cond_variances)

        return bald.squeeze(-1).squeeze(-1)

    def _compute_monte_carlo_information_gain(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        samples = self.get_posterior_samples(posterior)
        samples_log_prob = (
            posterior.mvn.log_prob(samples.squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        )
        breakpoint()
        normal = Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        return bald.squeeze(-1).squeeze(-1)


class StatisticalDistanceActiveLearning(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        X_pending: Optional[Tensor] = None,
        distance_metric: str = "hellinger",
        estimation_type: str = "LB",
        sampler: Optional[MCSampler] = None,
        num_samples: int = 64,
    ) -> None:
        """_summary_

        Args:
            model (Model): _description_
            X_pending (Optional[Tensor], optional): _description_. Defaults to None.
            distance_metric (str, optional): _description_. Defaults to "MC".
            sampler (Optional[MCSampler], optional): _description_. Defaults to None.
            num_samples (int, optional): _description_. Defaults to 64.

        Raises:
            ValueError: _description_
        """
        super().__init__(model)
        if estimation_type not in ["LB", "MC"]:
            raise ValueError(f"Estimation type {estimation_type} does not exist.")
        self.estimation_type = estimation_type
        self.set_X_pending(X_pending)
        # the default number of MC samples (512) are too many when doing FB modeling.
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        MCSamplerMixin.__init__(self, sampler=sampler)
        if distance_metric not in DISTANCE_METRICS.keys():
            raise ValueError(
                f"Distance metric need to be one of " f"{list(DISTANCE_METRICS.keys())}"
            )
        self.distance = DISTANCE_METRICS[distance_metric]

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        cond_means = posterior.mean
        marg_mean = cond_means.mean(dim=MCMC_DIM, keepdim=True)
        cond_variances = posterior.variance

        # the mixture variance is squeezed, need it unsqueezed
        marg_variance = posterior.mixture_variance.unsqueeze(MCMC_DIM)
        dist = self.distance(cond_means, marg_mean, cond_variances, marg_variance)
        # squeeze output dim and average over batch dim - MCMC dim is averaged laterc
        return dist.squeeze(-1).mean(-1)
