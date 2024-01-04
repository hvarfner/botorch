#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for Bayesian active learning.

References

.. [mackay1992alm]
    D. MacKay.
    Information-Based Objective Functions for Active Data Selection.
    Neural Computation, 1992.
.. [houlsby2011bald]
    N. Houlsby, F. Huszár, Z. Ghahramani, M. Lengyel.
    Bayesian Active Learning for Classification and Preference Learning.
    NIPS Workshop on Bayesian optimization, experimental design and bandits:
    Theory and applications, 2011.
.. [kirsch2011batchbald]
    Andreas Kirsch, Joost van Amersfoort, Yarin Gal.
    BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian
    Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2019.
.. [riis2022fbgp]
    C. Riis, F. Antunes, F. Hüttel, C. Azevedo, F. Pereira.
    Bayesian Active Learning with Fully Bayesian Gaussian Processes.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2022.
.. [Hvarfner2023scorebo]
    C. Hvarfner, F. Hutter, L. Nardi,
    Self-Correcting Bayesian Optimization thorugh Bayesian Active Learning.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2023.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.models.fully_bayesian import MCMC_DIM, SaasFullyBayesianSingleTaskGP
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.utils.stat_dist import mvn_hellinger_distance, mvn_kl_divergence
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor


SAMPLE_DIM = -4
DISTANCE_METRICS = {
    "hellinger": mvn_hellinger_distance,
    "kl_divergence": mvn_kl_divergence,
}


class FullyBayesianAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: Model):
        """Base class for acquisition functions which require a Fully Bayesian
        model treatment.

        Args:
            model: A fully bayesian single-outcome model.
        """
        if model._is_fully_bayesian:
            super().__init__(model)

        else:
            raise ValueError(
                "Fully Bayesian acquiition functions require "
                "a fully bayesian model (SaasFullyBayesianSingleTaskGP) to run."
            )


class qBayesianVarianceReduction(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        """Global variance reduction with fully Bayesian hyperparameter treatment by
        [mackay1992alm]_.

        Args:
            model: A fully bayesian single-outcome model.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points.
        """
        super().__init__(model)
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        res = torch.logdet(posterior.mixture_covariance_matrix).exp()

        # the MCMC dim is averaged out in the mixture postrior,
        # so the result needs to be unsqueeze[d for the averaging
        # in the decorator
        return res.unsqueeze(-1)


class qBayesianQueryByComittee(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        """
        Bayesian Query-By-Comittee [riis2022fbgp]_, which minimizes the variance
        of the mean in the posterior.

        Args:
            model: A fully bayesian single-outcome model.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
        """
        super().__init__(model)
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        posterior_mean = posterior.mean
        marg_mean = posterior.mixture_mean.unsqueeze(MCMC_DIM)
        mean_diff = posterior_mean - marg_mean
        covar_of_mean = torch.matmul(mean_diff, mean_diff.transpose(-1, -2))

        res = torch.logdet(covar_of_mean).exp()
        return torch.nan_to_num(res, 0)


class qBayesianActiveLearningByDisagreement(
    FullyBayesianAcquisitionFunction, MCSamplerMixin
):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "LB",
        sampler: Optional[MCSampler] = None,
        num_samples: int = 64,
    ) -> None:
        """Batch implementation [kirsch2019batchbald]_ of BALD [houlsby2011bald]_,
        which maximizes the mutual information between the next observation and the
        hyperparameters of the model.

        Args:
            model: A fully bayesian single-outcome model.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
            estimation_type: estimation_type: A string to determine which entropy
                estimate is computed: Lower bound" ("LB") or "Monte Carlo" ("MC")
                (not implemented yet).
            sampler: MCSampler for Monte Carlo estimation of the statistical distance
                (not implemented yet).
            num_samples (int, optional): Number of samples if employing monte carlo
                estimation of the statistical distance. Defaults to 64.
        """
        super().__init__(model)
        if estimation_type not in ["LB"]:
            raise ValueError(f"Estimation type {estimation_type} does not exist.")
        self.set_X_pending(X_pending)
        # the default number of MC samples (512) are too many when doing FB modeling.
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        MCSamplerMixin.__init__(self, sampler=sampler)

        if estimation_type == "LB":
            self.acq_method = self._compute_lower_bound_information_gain
        else:
            raise ValueError(
                "Only the 'LB' approximation to BALD is currently" "avaialable"
            )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        return self.acq_method(X)

    def _compute_lower_bound_information_gain(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        marg_covar = posterior.mixture_covariance_matrix
        cond_variances = posterior.variance

        prev_entropy = torch.logdet(marg_covar).unsqueeze(-1)
        # squeeze excess dim and mean over q-batch
        post_ub_entropy = torch.log(cond_variances).squeeze(-1).mean(-1)

        return prev_entropy - post_ub_entropy


class qStatisticalDistanceActiveLearning(FullyBayesianAcquisitionFunction):
    def __init__(
        self,
        model: SaasFullyBayesianSingleTaskGP,
        X_pending: Optional[Tensor] = None,
        distance_metric: str = "hellinger",
        estimation_type: str = "LB",
        sampler: Optional[MCSampler] = None,
        num_samples: int = 64,
    ) -> None:
        """Batch implementation of SAL [hvarfner2023scorebo]_, which minimizes
        discrepancy in the posterior predictive as measured by a statistical
        distance (or semi-metric).

        Args:
            model: A fully bayesian single-outcome model.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
            distance_metric (str, optional): The distance metric used. Defaults to
                "hellinger".
            estimation_type: estimation_type: A string to determine which entropy
                estimate is computed: Lower bound" ("LB") or "Monte Carlo" ("MC")
                (not implemented yet).
                Lower Bound is recommended due to the comparable empirical performance.
            sampler: MCSampler for Monte Carlo estimation of the statistical distance
                (not implemented yet).
            num_samples (int, optional): Number of samples if employing monte carlo
                estimation of the statistical distance. Defaults to 64.
        """
        super().__init__(model)
        # Currently only supports LB (lower bound) estimation, will add MC later on
        if estimation_type not in ["LB"]:
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

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        cond_means = posterior.mean
        marg_mean = posterior.mixture_mean.unsqueeze(MCMC_DIM)
        cond_covar = posterior.covariance_matrix

        # the mixture variance is squeezed, need it unsqueezed
        marg_covar = posterior.mixture_covariance_matrix.unsqueeze(MCMC_DIM)
        dist = self.distance(cond_means, marg_mean, cond_covar, marg_covar)

        # squeeze output dim - batch dim computed and reduced inside of dist
        # MCMC dim is averaged in decorator
        return dist.squeeze(-1)
