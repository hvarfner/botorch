#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A registry of helpers for generating inputs to acquisition function
constructors programmatically from a consistent input format.
"""

from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import (
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    LogConstrainedExpectedImprovement,
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    LogProbabilityOfImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.bayesian_active_learning import (
    qBayesianActiveLearningByDisagreement,
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
    qStatisticalDistanceActiveLearning,
)
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    TAU_MAX,
    TAU_RELU,
)
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective import (
    ExpectedHypervolumeImprovement,
    MCMultiOutputObjective,
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    AnalyticMultiOutputObjective,
    IdentityAnalyticMultiOutputObjective,
    IdentityMCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from botorch.acquisition.scorebo import qSelfCorrectingBayesianOptimization
from botorch.acquisition.utils import (
    compute_best_feasible_objective,
    expand_trace_observations,
    get_infeasible_cost,
    get_optimal_samples,
    project_to_target_fidelity,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.containers import BotorchContainer
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from torch import Tensor


ACQF_INPUT_CONSTRUCTOR_REGISTRY = {}

T = TypeVar("T")
MaybeDict = Union[T, Dict[Hashable, T]]
TOptimizeObjectiveKwargs = Union[
    None,
    MCAcquisitionObjective,
    PosteriorTransform,
    Tuple[Tensor, Tensor],
    Dict[int, float],
    bool,
    int,
    Dict[str, Any],
    Callable[[Tensor], Tensor],
    Tensor,
]


def _field_is_shared(
    datasets: Union[Iterable[SupervisedDataset], Dict[Hashable, SupervisedDataset]],
    fieldname: str,
) -> bool:
    r"""Determines whether or not a given field is shared by all datasets."""
    if isinstance(datasets, dict):
        datasets = datasets.values()

    base = None
    for dataset in datasets:
        if not hasattr(dataset, fieldname):
            raise AttributeError(f"{type(dataset)} object has no field `{fieldname}`.")

        obj = getattr(dataset, fieldname)
        if base is None:
            base = obj
        elif isinstance(base, Tensor):
            if not torch.equal(base, obj):
                return False
        elif base != obj:  # pragma: no cover
            return False

    return True


def _get_dataset_field(
    dataset: MaybeDict[SupervisedDataset],
    fieldname: str,
    transform: Optional[Callable[[BotorchContainer], Any]] = None,
    join_rule: Optional[Callable[[Sequence[Any]], Any]] = None,
    first_only: bool = False,
    assert_shared: bool = False,
) -> Any:
    r"""Convenience method for extracting a given field from one or more datasets."""
    if isinstance(dataset, dict):
        if assert_shared and not _field_is_shared(dataset, fieldname):
            raise ValueError(f"Field `{fieldname}` must be shared.")

        if not first_only:
            fields = (
                _get_dataset_field(d, fieldname, transform) for d in dataset.values()
            )
            return join_rule(tuple(fields)) if join_rule else tuple(fields)

        dataset = next(iter(dataset.values()))

    field = getattr(dataset, fieldname)
    return transform(field) if transform else field


def get_acqf_input_constructor(
    acqf_cls: Type[AcquisitionFunction],
) -> Callable[..., Dict[str, Any]]:
    r"""Get acquisition function input constructor from registry.

    Args:
        acqf_cls: The AcquisitionFunction class (not instance) for which
            to retrieve the input constructor.

    Returns:
        The input constructor associated with `acqf_cls`.

    """
    if acqf_cls not in ACQF_INPUT_CONSTRUCTOR_REGISTRY:
        raise RuntimeError(
            f"Input constructor for acquisition class `{acqf_cls.__name__}` not "
            "registered. Use the `@acqf_input_constructor` decorator to register "
            "a new method."
        )
    return ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls]


def allow_only_specific_variable_kwargs(f: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for allowing a function to accept keyword arguments that are not
    explicitly listed in the function signature, but only specific ones.

    This decorator is applied in `acqf_input_constructor` so that all constructors
    obtained with `acqf_input_constructor` allow keyword
    arguments such as `training_data` and `objective`, even if they do not appear
    in the signature of `f`. Any other keyword arguments will raise an error.
    """
    allowed = {
        "training_data",
        "objective",
        "posterior_transform",
        "X_baseline",
        "X_pending",
        "objective_thresholds",
        "constraints",
        "target_fidelities",
        "bounds",
    }

    def g(*args: Any, **kwargs: Any) -> T:
        new_kwargs = {}
        accepted_kwargs = inspect.signature(f).parameters.keys()
        for k, v in kwargs.items():
            if k in accepted_kwargs:
                new_kwargs[k] = v
            elif k not in allowed:
                raise TypeError(
                    f"Unexpected keyword argument `{k}` when"
                    f" constructing input arguments for {f.__name__}."
                )
        return f(*args, **new_kwargs)

    return g


def acqf_input_constructor(
    *acqf_cls: Type[AcquisitionFunction],
) -> Callable[..., AcquisitionFunction]:
    r"""Decorator for registering acquisition function input constructors.

    Args:
        acqf_cls: The AcquisitionFunction classes (not instances) for which
            to register the input constructor.
    """
    for acqf_cls_ in acqf_cls:
        if acqf_cls_ in ACQF_INPUT_CONSTRUCTOR_REGISTRY:
            raise ValueError(
                "Cannot register duplicate arg constructor for acquisition "
                f"class `{acqf_cls_.__name__}`"
            )

    def decorator(method):
        method_kwargs = allow_only_specific_variable_kwargs(method)
        for acqf_cls_ in acqf_cls:
            ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls_] = method_kwargs
        return method

    return decorator


def _register_acqf_input_constructor(
    acqf_cls: Type[AcquisitionFunction],
    input_constructor: Callable[..., Dict[str, Any]],
) -> None:
    ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls] = input_constructor


# --------------------- Input argument constructors --------------------- #


@acqf_input_constructor(PosteriorMean)
def construct_inputs_posterior_mean(
    model: Model,
    posterior_transform: Optional[PosteriorTransform] = None,
) -> Dict[str, Union[Model, Optional[PosteriorTransform]]]:
    r"""Construct kwargs for PosteriorMean acquisition function.

    Args:
        model: The model to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {"model": model, "posterior_transform": posterior_transform}


@acqf_input_constructor(
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    LogProbabilityOfImprovement,
)
def construct_inputs_best_f(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    posterior_transform: Optional[PosteriorTransform] = None,
    best_f: Optional[Union[float, Tensor]] = None,
    maximize: bool = True,
) -> Dict[str, Any]:
    r"""Construct kwargs for the acquisition functions requiring `best_f`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
            Used to determine default value for `best_f`.
        best_f: Threshold above (or below) which improvement is defined.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if best_f is None:
        best_f = get_best_f_analytic(
            training_data=training_data,
            posterior_transform=posterior_transform,
        )

    return {
        "model": model,
        "posterior_transform": posterior_transform,
        "best_f": best_f,
        "maximize": maximize,
    }


@acqf_input_constructor(UpperConfidenceBound)
def construct_inputs_ucb(
    model: Model,
    posterior_transform: Optional[PosteriorTransform] = None,
    beta: Union[float, Tensor] = 0.2,
    maximize: bool = True,
) -> Dict[str, Any]:
    r"""Construct kwargs for `UpperConfidenceBound`.

    Args:
        model: The model to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the trade-off parameter between mean and covariance
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        "model": model,
        "posterior_transform": posterior_transform,
        "beta": beta,
        "maximize": maximize,
    }


@acqf_input_constructor(
    ConstrainedExpectedImprovement, LogConstrainedExpectedImprovement
)
def construct_inputs_constrained_ei(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective_index: int,
    constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
    maximize: bool = True,
) -> Dict[str, Any]:
    r"""Construct kwargs for `ConstrainedExpectedImprovement`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective_index: The index of the objective.
        constraints: A dictionary of the form `{i: [lower, upper]}`, where
            `i` is the output index, and `lower` and `upper` are lower and upper
            bounds on that output (resp. interpreted as -Inf / Inf if None)
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    # TODO: Implement best point computation from training data
    # best_f =
    # return {
    #     "model": model,
    #     "best_f": best_f,
    #     "objective_index": objective_index,
    #     "constraints": constraints,
    #     "maximize": maximize,
    # }
    raise NotImplementedError  # pragma: nocover


@acqf_input_constructor(NoisyExpectedImprovement, LogNoisyExpectedImprovement)
def construct_inputs_noisy_ei(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    num_fantasies: int = 20,
    maximize: bool = True,
) -> Dict[str, Any]:
    r"""Construct kwargs for `NoisyExpectedImprovement`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        num_fantasies: The number of fantasies to generate. The higher this
            number the more accurate the model (at the expense of model
            complexity and performance).
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    # TODO: Add prune_baseline functionality as for qNEI
    X = _get_dataset_field(training_data, "X", first_only=True, assert_shared=True)
    return {
        "model": model,
        "X_observed": X,
        "num_fantasies": num_fantasies,
        "maximize": maximize,
    }


@acqf_input_constructor(qSimpleRegret)
def construct_inputs_qSimpleRegret(
    model: Model,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    X_baseline: Optional[Tensor] = None,
) -> Dict[str, Any]:
    r"""Construct kwargs for qSimpleRegret.

    Args:
        model: The model to be used in the acquisition function.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
            that have points that have been submitted for function evaluation
            but have not yet been evaluated.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension `sample_shape x batch-shape x q x m`-dim to a
            `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, checks that all
            training_data have the same input features and take the first `X`.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if constraints is not None:
        if X_baseline is None:
            raise ValueError("Constraints require an X_baseline.")
        objective = ConstrainedMCObjective(
            objective=objective,
            constraints=constraints,
            infeasible_cost=get_infeasible_cost(
                X=X_baseline, model=model, objective=objective
            ),
        )
    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
    }


@acqf_input_constructor(qExpectedImprovement)
def construct_inputs_qEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    best_f: Optional[Union[float, Tensor]] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    eta: Union[Tensor, float] = 1e-3,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qExpectedImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        best_f: Threshold above (or below) which improvement is defined.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension `sample_shape x batch-shape x q x m`-dim to a
            `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of `compute_smoothed_feasibility_indicator`.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if best_f is None:
        best_f = get_best_f_mc(
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            constraints=constraints,
            model=model,
        )

    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
        "best_f": best_f,
        "constraints": constraints,
        "eta": eta,
    }


@acqf_input_constructor(qLogExpectedImprovement)
def construct_inputs_qLogEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    best_f: Optional[Union[float, Tensor]] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    eta: Union[Tensor, float] = 1e-3,
    fat: bool = True,
    tau_max: float = TAU_MAX,
    tau_relu: float = TAU_RELU,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qExpectedImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        best_f: Threshold above (or below) which improvement is defined.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension `sample_shape x batch-shape x q x m`-dim to a
            `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of `compute_smoothed_feasibility_indicator`.
        fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
            approximation to the ReLU.
        tau_max: Temperature parameter controlling the sharpness of the smooth
            approximations to max.
        tau_relu: Temperature parameter controlling the sharpness of the smooth
            approximations to ReLU.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        **construct_inputs_qEI(
            model=model,
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sampler=sampler,
            best_f=best_f,
            constraints=constraints,
            eta=eta,
        ),
        "fat": fat,
        "tau_max": tau_max,
        "tau_relu": tau_relu,
    }


@acqf_input_constructor(qNoisyExpectedImprovement)
def construct_inputs_qNEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    X_baseline: Optional[Tensor] = None,
    prune_baseline: Optional[bool] = True,
    cache_root: Optional[bool] = True,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    eta: Union[Tensor, float] = 1e-3,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qNoisyExpectedImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, checks that all
            training_data have the same input features and take the first `X`.
        prune_baseline: If True, remove points in `X_baseline` that are
            highly unlikely to be the best point. This can significantly
            improve performance and is generally recommended.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension `sample_shape x batch-shape x q x m`-dim to a
            `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of `compute_smoothed_feasibility_indicator`.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if X_baseline is None:
        X_baseline = _get_dataset_field(
            training_data,
            fieldname="X",
            assert_shared=True,
            first_only=True,
        )
    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
        "X_baseline": X_baseline,
        "prune_baseline": prune_baseline,
        "cache_root": cache_root,
        "constraints": constraints,
        "eta": eta,
    }


@acqf_input_constructor(qLogNoisyExpectedImprovement)
def construct_inputs_qLogNEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    X_baseline: Optional[Tensor] = None,
    prune_baseline: Optional[bool] = True,
    cache_root: Optional[bool] = True,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    eta: Union[Tensor, float] = 1e-3,
    fat: bool = True,
    tau_max: float = TAU_MAX,
    tau_relu: float = TAU_RELU,
):
    r"""Construct kwargs for the `qLogNoisyExpectedImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, checks that all
            training_data have the same input features and take the first `X`.
        prune_baseline: If True, remove points in `X_baseline` that are
            highly unlikely to be the best point. This can significantly
            improve performance and is generally recommended.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension `sample_shape x batch-shape x q x m`-dim to a
            `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of `compute_smoothed_feasibility_indicator`.
        fat: Toggles the use of the fat-tailed non-linearities to smoothly approximate
            the constraints indicator function.
        tau_max: Temperature parameter controlling the sharpness of the smooth
            approximations to max.
        tau_relu: Temperature parameter controlling the sharpness of the smooth
            approximations to ReLU.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        **construct_inputs_qNEI(
            model=model,
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sampler=sampler,
            X_baseline=X_baseline,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
            constraints=constraints,
            eta=eta,
        ),
        "fat": fat,
        "tau_max": tau_max,
        "tau_relu": tau_relu,
    }


@acqf_input_constructor(qProbabilityOfImprovement)
def construct_inputs_qPI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    tau: float = 1e-3,
    best_f: Optional[Union[float, Tensor]] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    eta: Union[Tensor, float] = 1e-3,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qProbabilityOfImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        tau: The temperature parameter used in the sigmoid approximation
            of the step function. Smaller values yield more accurate
            approximations of the function, but result in gradients
            estimates with higher variance.
        best_f: The best objective value observed so far (assumed noiseless). Can
            be a `batch_shape`-shaped tensor, which in case of a batched model
            specifies potentially different values for each element of the batch.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension `sample_shape x batch-shape x q x m`-dim to a
            `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of `compute_smoothed_feasibility_indicator`.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if best_f is None:
        best_f = get_best_f_mc(
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            constraints=constraints,
            model=model,
        )

    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
        "tau": tau,
        "best_f": best_f,
        "constraints": constraints,
        "eta": eta,
    }


@acqf_input_constructor(qUpperConfidenceBound)
def construct_inputs_qUCB(
    model: Model,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    beta: float = 0.2,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qUpperConfidenceBound` constructor.

    Args:
        model: The model to be used in the acquisition function.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        beta: Controls tradeoff between mean and standard deviation in UCB.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
        "beta": beta,
    }


def _get_sampler(mc_samples: int, qmc: bool) -> MCSampler:
    """Set up MC sampler for q(N)EHVI."""
    # initialize the sampler
    shape = torch.Size([mc_samples])
    if qmc:
        return SobolQMCNormalSampler(sample_shape=shape)
    return IIDNormalSampler(sample_shape=shape)


@acqf_input_constructor(ExpectedHypervolumeImprovement)
def construct_inputs_EHVI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective_thresholds: Tensor,
    objective: Optional[AnalyticMultiOutputObjective] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    alpha: Optional[float] = None,
    Y_pmean: Optional[Tensor] = None,
) -> Dict[str, Any]:
    r"""Construct kwargs for `ExpectedHypervolumeImprovement` constructor."""
    num_objectives = objective_thresholds.shape[0]
    if constraints is not None:
        raise NotImplementedError("EHVI does not yet support outcome constraints.")

    X = _get_dataset_field(
        training_data,
        fieldname="X",
        first_only=True,
        assert_shared=True,
    )

    alpha = (
        get_default_partitioning_alpha(num_objectives=num_objectives)
        if alpha is None
        else alpha
    )
    # This selects the objectives (a subset of the outcomes) and set each
    # objective threshold to have the proper optimization direction.
    if objective is None:
        objective = IdentityAnalyticMultiOutputObjective()
    if isinstance(objective, RiskMeasureMCObjective):
        pre_obj = objective.preprocessing_function
    else:
        pre_obj = objective
    ref_point = pre_obj(objective_thresholds)

    # Compute posterior mean (for ref point computation ref pareto frontier)
    # if one is not provided among arguments.
    if Y_pmean is None:
        with torch.no_grad():
            Y_pmean = model.posterior(X).mean
    if alpha > 0:
        partitioning = NondominatedPartitioning(
            ref_point=ref_point,
            Y=pre_obj(Y_pmean),
            alpha=alpha,
        )
    else:
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=pre_obj(Y_pmean),
        )

    return {
        "model": model,
        "ref_point": ref_point,
        "partitioning": partitioning,
        "objective": objective,
    }


@acqf_input_constructor(qExpectedHypervolumeImprovement)
def construct_inputs_qEHVI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective_thresholds: Tensor,
    objective: Optional[MCMultiOutputObjective] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    alpha: Optional[float] = None,
    sampler: Optional[MCSampler] = None,
    X_pending: Optional[Tensor] = None,
    eta: float = 1e-3,
    mc_samples: int = 128,
    qmc: bool = True,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qExpectedHypervolumeImprovement` constructor."""
    X = _get_dataset_field(
        training_data,
        fieldname="X",
        first_only=True,
        assert_shared=True,
    )

    # compute posterior mean (for ref point computation ref pareto frontier)
    with torch.no_grad():
        Y_pmean = model.posterior(X).mean
    # For HV-based acquisition functions we pass the constraint transform directly
    if constraints is not None:
        # Adjust `Y_pmean` to contain feasible points only.
        feas = torch.stack([c(Y_pmean) <= 0 for c in constraints], dim=-1).all(dim=-1)
        Y_pmean = Y_pmean[feas]

    if objective is None:
        objective = IdentityMCMultiOutputObjective()

    ehvi_kwargs = construct_inputs_EHVI(
        model=model,
        training_data=training_data,
        objective_thresholds=objective_thresholds,
        objective=objective,
        constraints=None,
        alpha=alpha,
        # Pass `Y_pmean` that accounts for constraints to `construct_inputs_EHVI`
        # to ensure that correct non-dominated partitioning is produced.
        Y_pmean=Y_pmean,
    )

    if sampler is None and isinstance(model, GPyTorchModel):
        sampler = _get_sampler(mc_samples=mc_samples, qmc=qmc)

    add_qehvi_kwargs = {
        "sampler": sampler,
        "X_pending": X_pending,
        "constraints": constraints,
        "eta": eta,
    }
    return {**ehvi_kwargs, **add_qehvi_kwargs}


@acqf_input_constructor(qNoisyExpectedHypervolumeImprovement)
def construct_inputs_qNEHVI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective_thresholds: Tensor,
    objective: Optional[MCMultiOutputObjective] = None,
    X_baseline: Optional[Tensor] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    alpha: Optional[float] = None,
    sampler: Optional[MCSampler] = None,
    X_pending: Optional[Tensor] = None,
    eta: float = 1e-3,
    fat: bool = False,
    mc_samples: int = 128,
    qmc: bool = True,
    prune_baseline: bool = True,
    cache_pending: bool = True,
    max_iep: int = 0,
    incremental_nehvi: bool = True,
    cache_root: bool = True,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qNoisyExpectedHypervolumeImprovement`'s constructor."""
    if X_baseline is None:
        X_baseline = _get_dataset_field(
            training_data,
            fieldname="X",
            first_only=True,
            assert_shared=True,
        )
    # This selects the objectives (a subset of the outcomes) and set each
    # objective threshold to have the proper optimization direction.
    if objective is None:
        objective = IdentityMCMultiOutputObjective()

    if constraints is not None:
        if isinstance(objective, RiskMeasureMCObjective):
            raise UnsupportedError(
                "Outcome constraints are not supported with risk measures. "
                "Use a feasibility-weighted risk measure instead."
            )

    if sampler is None and isinstance(model, GPyTorchModel):
        sampler = _get_sampler(mc_samples=mc_samples, qmc=qmc)

    if isinstance(objective, RiskMeasureMCObjective):
        ref_point = objective.preprocessing_function(objective_thresholds)
    else:
        ref_point = objective(objective_thresholds)

    num_objectives = objective_thresholds[~torch.isnan(objective_thresholds)].shape[0]
    if alpha is None:
        alpha = get_default_partitioning_alpha(num_objectives=num_objectives)

    return {
        "model": model,
        "ref_point": ref_point,
        "X_baseline": X_baseline,
        "sampler": sampler,
        "objective": objective,
        "constraints": constraints,
        "X_pending": X_pending,
        "eta": eta,
        "fat": fat,
        "prune_baseline": prune_baseline,
        "alpha": alpha,
        "cache_pending": cache_pending,
        "max_iep": max_iep,
        "incremental_nehvi": incremental_nehvi,
        "cache_root": cache_root,
    }


@acqf_input_constructor(qLogNoisyExpectedHypervolumeImprovement)
def construct_inputs_qLogNEHVI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective_thresholds: Tensor,
    objective: Optional[MCMultiOutputObjective] = None,
    X_baseline: Optional[Tensor] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    alpha: Optional[float] = None,
    sampler: Optional[MCSampler] = None,
    X_pending: Optional[Tensor] = None,
    eta: float = 1e-3,
    fat: bool = True,
    mc_samples: int = 128,
    qmc: bool = True,
    prune_baseline: bool = True,
    cache_pending: bool = True,
    max_iep: int = 0,
    incremental_nehvi: bool = True,
    cache_root: bool = True,
    tau_relu: float = TAU_RELU,
    tau_max: float = TAU_MAX,
) -> Dict[str, Any]:
    """
    Construct kwargs for `qLogNoisyExpectedHypervolumeImprovement`'s constructor."
    """
    return {
        **construct_inputs_qNEHVI(
            model=model,
            training_data=training_data,
            objective_thresholds=objective_thresholds,
            objective=objective,
            X_baseline=X_baseline,
            constraints=constraints,
            alpha=alpha,
            sampler=sampler,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            mc_samples=mc_samples,
            qmc=qmc,
            prune_baseline=prune_baseline,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
        ),
        "tau_relu": tau_relu,
        "tau_max": tau_max,
    }


@acqf_input_constructor(qMaxValueEntropy)
def construct_inputs_qMES(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    candidate_size: int = 1000,
    maximize: bool = True,
    # TODO: qMES also supports other inputs, such as num_fantasies
) -> Dict[str, Any]:
    r"""Construct kwargs for `qMaxValueEntropy` constructor."""

    X = _get_dataset_field(training_data, "X", first_only=True)
    _kw = {"device": X.device, "dtype": X.dtype}
    _rvs = torch.rand(candidate_size, len(bounds), **_kw)
    _bounds = torch.as_tensor(bounds, **_kw).transpose(0, 1)
    return {
        "model": model,
        "candidate_set": _bounds[0] + (_bounds[1] - _bounds[0]) * _rvs,
        "maximize": maximize,
    }


def construct_inputs_mf_base(
    target_fidelities: Dict[int, Union[int, float]],
    fidelity_weights: Optional[Dict[int, float]] = None,
    cost_intercept: float = 1.0,
    num_trace_observations: int = 0,
) -> Dict[str, Any]:
    r"""Construct kwargs for a multifidelity acquisition function's constructor."""
    if fidelity_weights is None:
        fidelity_weights = {f: 1.0 for f in target_fidelities}

    if set(target_fidelities) != set(fidelity_weights):
        raise RuntimeError(
            "Must provide the same indices for target_fidelities "
            f"({set(target_fidelities)}) and fidelity_weights "
            f" ({set(fidelity_weights)})."
        )

    cost_aware_utility = InverseCostWeightedUtility(
        cost_model=AffineFidelityCostModel(
            fidelity_weights=fidelity_weights, fixed_cost=cost_intercept
        )
    )

    return {
        "cost_aware_utility": cost_aware_utility,
        "expand": lambda X: expand_trace_observations(
            X=X,
            fidelity_dims=sorted(target_fidelities),
            num_trace_obs=num_trace_observations,
        ),
        "project": lambda X: project_to_target_fidelity(
            X=X, target_fidelities=target_fidelities
        ),
    }


@acqf_input_constructor(qKnowledgeGradient)
def construct_inputs_qKG(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    num_fantasies: int = 64,
    **optimize_objective_kwargs: TOptimizeObjectiveKwargs,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qKnowledgeGradient` constructor."""

    X = _get_dataset_field(training_data, "X", first_only=True)
    _bounds = torch.as_tensor(bounds, dtype=X.dtype, device=X.device)

    _, current_value = optimize_objective(
        model=model,
        bounds=_bounds.t(),
        q=1,
        objective=objective,
        posterior_transform=posterior_transform,
        **optimize_objective_kwargs,
    )

    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "num_fantasies": num_fantasies,
        "current_value": current_value.detach().cpu().max(),
    }


@acqf_input_constructor(qMultiFidelityKnowledgeGradient)
def construct_inputs_qMFKG(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    target_fidelities: Dict[int, Union[int, float]],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    fidelity_weights: Optional[Dict[int, float]] = None,
    cost_intercept: float = 1.0,
    num_trace_observations: int = 0,
    num_fantasies: int = 64,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qMultiFidelityKnowledgeGradient` constructor."""

    inputs_mf = construct_inputs_mf_base(
        target_fidelities=target_fidelities,
        fidelity_weights=fidelity_weights,
        cost_intercept=cost_intercept,
        num_trace_observations=num_trace_observations,
    )

    inputs_kg = construct_inputs_qKG(
        model=model,
        training_data=training_data,
        bounds=bounds,
        objective=objective,
        posterior_transform=posterior_transform,
        num_fantasies=num_fantasies,
    )

    return {**inputs_mf, **inputs_kg}


@acqf_input_constructor(qMultiFidelityMaxValueEntropy)
def construct_inputs_qMFMES(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    target_fidelities: Dict[int, Union[int, float]],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    num_fantasies: int = 64,
    X_baseline: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    objective_thresholds: Optional[Tensor] = None,
    outcome_constraints: Optional[List[Tuple[Tensor, Tensor]]] = None,
    fidelity_weights: Optional[Dict[int, float]] = None,
    cost_intercept: float = 1.0,
    num_trace_observations: int = 0,
    candidate_size: int = 1000,
    maximize: bool = True,
    **optimize_objective_kwargs: TOptimizeObjectiveKwargs,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qMultiFidelityMaxValueEntropy` constructor."""
    inputs_mf = construct_inputs_mf_base(
        target_fidelities=target_fidelities,
        fidelity_weights=fidelity_weights,
        cost_intercept=cost_intercept,
        num_trace_observations=num_trace_observations,
    )

    inputs_qmes = construct_inputs_qMES(
        model=model,
        training_data=training_data,
        bounds=bounds,
        candidate_size=candidate_size,
        maximize=maximize,
    )

    X = _get_dataset_field(training_data, "X", first_only=True)
    _bounds = torch.as_tensor(bounds, dtype=X.dtype, device=X.device)
    _, current_value = optimize_objective(
        model=model,
        bounds=_bounds.t(),
        q=1,
        objective=objective,
        posterior_transform=posterior_transform,
        **optimize_objective_kwargs,
    )

    return {
        **inputs_mf,
        **inputs_qmes,
        "current_value": current_value.detach().cpu().max(),
        "target_fidelities": target_fidelities,
    }


@acqf_input_constructor(AnalyticExpectedUtilityOfBestOption)
def construct_inputs_analytic_eubo(
    model: Model,
    pref_model: Model,
    previous_winner: Optional[Tensor] = None,
    sample_multiplier: Optional[float] = 1.0,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `AnalyticExpectedUtilityOfBestOption` constructor.

    Args:
        model: The outcome model to be used in the acquisition function.
        pref_model: The preference model to be used in preference exploration.
        previous_winner: The previous winner of the best option.
        sample_multiplier: The scale factor for the single-sample model.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    # construct a deterministic fixed single sample model from `model`
    # i.e., performing EUBO-zeta by default as described
    # in https://arxiv.org/abs/2203.11382
    # using pref_model.dim instead of model.num_outputs here as MTGP's
    # num_outputs could be tied to the number of tasks
    w = torch.randn(pref_model.dim) * sample_multiplier
    one_sample_outcome_model = FixedSingleSampleModel(model=model, w=w)

    return {
        "pref_model": pref_model,
        "outcome_model": one_sample_outcome_model,
        "previous_winner": previous_winner,
    }


def get_best_f_analytic(
    training_data: MaybeDict[SupervisedDataset],
    posterior_transform: Optional[PosteriorTransform] = None,
) -> Tensor:
    if isinstance(training_data, dict) and not _field_is_shared(
        training_data, fieldname="X"
    ):
        raise NotImplementedError("Currently only block designs are supported.")

    Y = _get_dataset_field(
        training_data,
        fieldname="Y",
        join_rule=lambda field_tensors: torch.cat(field_tensors, dim=-1),
    )

    if posterior_transform is not None:
        return posterior_transform.evaluate(Y).max(-1).values
    if Y.shape[-1] > 1:
        raise NotImplementedError(
            "Analytic acquisition functions currently only work with "
            "multi-output models if provided with a `ScalarizedObjective`."
        )
    return Y.max(-2).values.squeeze(-1)


def get_best_f_mc(
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    model: Optional[Model] = None,
) -> Tensor:
    """
    Computes the maximum value of the objective over the training data.

    Args:
        training_data: Has fields Y, which is evaluated by `objective`, and X,
            which is used as `X_baseline`. `Y` is of shape
            `batch_shape x q x m`.
        objective: The objective under which to evaluate the training data. If
            omitted, uses `IdentityMCObjective`.
        posterior_transform: An optional PosteriorTransform to apply to `Y`
            before computing the objective.
        constraints: For assessing feasibility.
        model: Used by `compute_best_feasible_objective` when there are no
            feasible observations.

    Returns:
        A Tensor of shape `batch_shape`.
    """
    if isinstance(training_data, dict) and not _field_is_shared(
        training_data, fieldname="X"
    ):
        raise NotImplementedError("Currently only block designs are supported.")

    X_baseline = _get_dataset_field(
        training_data,
        fieldname="X",
        assert_shared=True,
        first_only=True,
    )

    Y = _get_dataset_field(
        training_data,
        fieldname="Y",
        join_rule=lambda field_tensors: torch.cat(field_tensors, dim=-1),
    )  # batch_shape x q x m

    if posterior_transform is not None:
        # retain the original tensor dimension since objective expects explicit
        # output dimension.
        Y_dim = Y.dim()
        Y = posterior_transform.evaluate(Y)
        if Y.dim() < Y_dim:
            Y = Y.unsqueeze(-1)
    if objective is None:
        if Y.shape[-1] > 1:
            raise UnsupportedError(
                "Acquisition functions require an objective when "
                "used with multi-output models (except for multi-objective"
                "acquisition functions)."
            )
        objective = IdentityMCObjective()
    # `Y` is of shape `(batch_shape) x q x m`; `MCAcquisitionObjective`s expect
    # inputs `sample_shape x (batch_shape) x q x m`.
    # For most objectives, `obj` will have shape `1 x (batch_shape) x q`, but
    # with a `LearnedObjective` it can be `num_samples x (batch_shape) x q`.
    obj = objective(Y.unsqueeze(0), X=X_baseline)
    if obj.shape[0] > 1:
        obj = obj.mean(dim=0)
    else:
        obj = obj.squeeze(0)
    return compute_best_feasible_objective(
        samples=Y,
        obj=obj,
        constraints=constraints,
        model=model,
        objective=objective,
        posterior_transform=posterior_transform,
        X_baseline=X_baseline,
    ).squeeze()


def optimize_objective(
    model: Model,
    bounds: Tensor,
    q: int,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    qmc: bool = True,
    mc_samples: int = 512,
    seed_inner: Optional[int] = None,
    optimizer_options: Optional[Dict[str, Any]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    sequential: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Optimize an objective under the given model.

    Args:
        model: The model to be used in the objective.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The cardinality of input sets on which the objective is to be evaluated.
        objective: The objective to optimize.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        linear_constraints: A tuple of (A, b). Given `k` linear constraints on a
            `d`-dimensional space, `A` is `k x d` and `b` is `k x 1` such that
            `A x <= b`. (Not used by single task models).
        fixed_features: A dictionary of feature assignments `{feature_index: value}` to
            hold fixed during generation.
        qmc: Toggle for enabling (qmc=1) or disabling (qmc=0) use of Quasi Monte Carlo.
        mc_samples: Integer number of samples used to estimate Monte Carlo objectives.
        seed_inner: Integer seed used to initialize the sampler passed to MCObjective.
        optimizer_options: Table used to lookup keyword arguments for the optimizer.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e. according to `round-trip` transformations).
        batch_initial_conditions: A Tensor of initial values for the optimizer.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.

    Returns:
        A tuple containing the best input locations and corresponding objective values.
    """
    if optimizer_options is None:
        optimizer_options = {}

    if objective is not None:
        sampler_cls = SobolQMCNormalSampler if qmc else IIDNormalSampler
        acq_function = qSimpleRegret(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            sampler=sampler_cls(sample_shape=torch.Size([mc_samples]), seed=seed_inner),
        )
    else:
        acq_function = PosteriorMean(
            model=model, posterior_transform=posterior_transform
        )

    if fixed_features:
        acq_function = FixedFeatureAcquisitionFunction(
            acq_function=acq_function,
            d=bounds.shape[-1],
            columns=list(fixed_features.keys()),
            values=list(fixed_features.values()),
        )
        free_feature_dims = list(range(len(bounds)) - fixed_features.keys())
        free_feature_bounds = bounds[:, free_feature_dims]  # (2, d' <= d)
    else:
        free_feature_bounds = bounds

    if linear_constraints is None:
        inequality_constraints = None
    else:
        A, b = linear_constraints
        inequality_constraints = []
        k, d = A.shape
        for i in range(k):
            indices = A[i, :].nonzero(as_tuple=False).squeeze()
            coefficients = -A[i, indices]
            rhs = -b[i, 0]
            inequality_constraints.append((indices, coefficients, rhs))

    return optimize_acqf(
        acq_function=acq_function,
        bounds=free_feature_bounds,
        q=q,
        num_restarts=optimizer_options.get("num_restarts", 60),
        raw_samples=optimizer_options.get("raw_samples", 1024),
        options={
            "batch_limit": optimizer_options.get("batch_limit", 8),
            "maxiter": optimizer_options.get("maxiter", 200),
            "nonnegative": optimizer_options.get("nonnegative", False),
            "method": optimizer_options.get("method", "L-BFGS-B"),
        },
        inequality_constraints=inequality_constraints,
        fixed_features=None,  # handled inside the acquisition function
        post_processing_func=post_processing_func,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=True,
        sequential=sequential,
    )


@acqf_input_constructor(qJointEntropySearch)
def construct_inputs_qJES(
    model: Model,
    bounds: List[Tuple[float, float]],
    num_optima: int = 64,
    maximize: bool = True,
    condition_noiseless: bool = True,
    X_pending: Optional[Tensor] = None,
    estimation_type: str = "LB",
    num_samples: int = 64,
):
    dtype = model.train_targets.dtype
    optimal_inputs, optimal_outputs = get_optimal_samples(
        model=model,
        bounds=torch.as_tensor(bounds, dtype=dtype).T,
        num_optima=num_optima,
        maximize=maximize,
    )

    inputs = {
        "model": model,
        "optimal_inputs": optimal_inputs,
        "optimal_outputs": optimal_outputs,
        "condition_noiseless": condition_noiseless,
        "maximize": maximize,
        "X_pending": X_pending,
        "estimation_type": estimation_type,
        "num_samples": num_samples,
    }
    return inputs


@acqf_input_constructor(
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
)
def construct_inputs_BAL(
    model: Model,
    X_pending: Optional[Tensor] = None,
):
    inputs = {
        "model": model,
        "X_pending": X_pending,
    }
    return inputs


@acqf_input_constructor(qBayesianActiveLearningByDisagreement)
def construct_inputs_BALD(
    model: Model,
    X_pending: Optional[Tensor] = None,
    estimation_type: Optional[str] = "LB",
    num_samples: Optional[int] = 64,
):
    inputs = {
        "model": model,
        "X_pending": X_pending,
        "estimation_type": estimation_type,
        "num_samples": num_samples,
    }
    return inputs


@acqf_input_constructor(qStatisticalDistanceActiveLearning)
def construct_inputs_SAL(
    model: Model,
    distance_metric: str = "hellinger",
    X_pending: Optional[Tensor] = None,
    estimation_type: str = "LB",
    num_samples: int = 64,
):
    inputs = {
        "model": model,
        "distance_metric": distance_metric,
        "X_pending": X_pending,
        "estimation_type": estimation_type,
        "num_samples": num_samples,
    }
    return inputs


@acqf_input_constructor(qSelfCorrectingBayesianOptimization)
def construct_inputs_SCoreBO(
    model: Model,
    bounds: List[Tuple[float, float]],
    num_optima: int = 8,
    maximize: bool = True,
    distance_metric: str = "hellinger",
    X_pending: Optional[Tensor] = None,
    estimation_type: str = "LB",
    num_samples: int = 64,
):
    dtype = model.train_targets.dtype
    # the number of optima are per model
    optimal_inputs, optimal_outputs = get_optimal_samples(
        model=model,
        bounds=torch.as_tensor(bounds, dtype=dtype).T,
        num_optima=num_optima,
        maximize=maximize,
    )

    inputs = {
        "model": model,
        "optimal_inputs": optimal_inputs,
        "optimal_outputs": optimal_outputs,
        "distance_metric": distance_metric,
        "maximize": maximize,
        "X_pending": X_pending,
        "estimation_type": estimation_type,
        "num_samples": num_samples,
    }
    return inputs
