
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Callable, List, Optional, Protocol, Tuple, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.utils import (
    compute_best_feasible_objective,
    prune_inferior_points,
    repeat_to_match_aug_dim,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor

class qDistanceWeightedImprovementOverThreshold(AcquisitionFunction, MCSamplerMixin): #qDWIT
    r"""MC-based batch Probability of Improvement.

    Estimates the probability of improvement over the current best observed
    value by sampling from the joint posterior distribution of the q-batch.
    MC-based estimates of a probability involves taking expectation of an
    indicator function; to support auto-differentiation, the indicator is
    replaced with a sigmoid function with temperature parameter `tau`.

    `qPI(X) = P(max Y >= best_f), Y ~ f(X), X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qPI = qProbabilityOfImprovement(model, best_f, sampler)
        >>> qpi = qPI(test_X)
    """

    def __init__(
        self,
        model: Model,
        objective_thresholds: Optional[Union[float, Tensor]] = None,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
        tau: float = 1e-3,
        eta: Union[Tensor, float] = 1e-3,
    ) -> None:
        r"""q-Probability of Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can
                be a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
                NOTE: `ConstrainedMCObjective` for outcome constraints is deprecated in
                favor of passing the `constraints` directly to this constructor.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.  Concatenated into X upon
                forward call.  Copied and set to have no gradient.
            tau: The temperature parameter used in the sigmoid approximation
                of the step function. Smaller values yield more accurate
                approximations of the function, but result in gradients
                estimates with higher variance.
            constraints: A list of constraint callables which map posterior samples to
                a scalar. The associated constraint is considered satisfied if this
                scalar is less than zero.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. For more details, on this
                parameter, see the docs of `compute_smoothed_feasibility_indicator`.
        """
        super().__init__(model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)

        if objective is None:
            objective = IdentityMCObjective()
        self.objective: MCAcquisitionObjective = objective
        self.set_X_pending(X_pending)

        reference = torch.as_tensor(objective_thresholds, dtype=float)  # adding batch dim
        self.register_buffer("reference", reference)
        self.register_buffer("tau", torch.as_tensor(tau, dtype=float))
        training_data = model.train_inputs[0]
        if training_data.ndim == 3: 
            # if multiobjective, we want to reduce the (identical sets of) training data
            training_data = training_data[0]

        better_than_ref = torch.all(model.train_targets > self.reference.unsqueeze(-1), dim=0)
        self.X_baseline = training_data[better_than_ref]
        print(self.X_baseline)
        
    def _get_samples_and_objectives(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes posterior samples and objective values at input X.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of model inputs.

        Returns:
            A two-tuple `(samples, obj)`, where `samples` is a tensor of posterior
            samples with shape `sample_shape x batch_shape x q x m`, and `obj` is a
            tensor of MC objective values with shape `sample_shape x batch_shape x q`.
        """
        posterior = self.model.posterior(X=X)
        samples = self.get_posterior_samples(posterior)
        return samples, samples

    #@concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qProbabilityOfImprovement per sample on the candidate set `X`.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of improvement indicators.
        """
        if self.X_pending is not None:
            baseline = torch.cat((self.X_baseline, self.X_pending))
        else: 
            baseline = self.X_baseline
    
        _, obj = self._get_samples_and_objectives(X)
        improvement = obj - self.reference.to(obj)
        prob_imp = torch.sigmoid(improvement / self.tau).prod(dim=-1)
        
        if len(baseline) == 0:
            # mean over MC dim and sum over q-batch
            return prob_imp.mean(0).sum(-1)
        
        dist = torch.pow(torch.norm(X - baseline, p=2, dim=-1).min(dim=-1, keepdim=True).values, 2)
        return (prob_imp * dist).mean(0).sum(-1)

