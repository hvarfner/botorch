#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from botorch.acquisition.scorebo import SelfCorrectingBayesianOptimization
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase





def _get_mcmc_samples(num_samples: int, dim: int, infer_noise: bool, **tkwargs):

    mcmc_samples = {
        "lengthscale": torch.rand(num_samples, 1, dim, **tkwargs),
        "outputscale": torch.rand(num_samples, **tkwargs),
        "mean": torch.randn(num_samples, **tkwargs),
    }
    if infer_noise:
        mcmc_samples["noise"] = torch.rand(num_samples, 1, **tkwargs)
    return mcmc_samples

def get_model(
        train_X, 
        train_Y, 
        num_models, 
        use_model_list, 
        standardize_model, 
        infer_noise,
        **tkwargs,

):
    num_objectives = train_Y.shape[-1]

    if standardize_model:
        if use_model_list:
            outcome_transform = Standardize(m=1)
        else:
            outcome_transform = Standardize(m=num_objectives)
    else:
        outcome_transform = None
    
    mcmc_samples = _get_mcmc_samples(
        num_samples=num_models,
        dim=train_X.shape[-1],
        infer_noise=infer_noise,
        **tkwargs,
    )

    if use_model_list:
        gp1 = SaasFullyBayesianSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y[:, i : i + 1],
            outcome_transform=outcome_transform,
        )
        gp1.load_mcmc_samples(mcmc_samples)
        gp2 = SaasFullyBayesianSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y[:, i : i + 1],
            outcome_transform=outcome_transform,
        )
        gp2.load_mcmc_samples(mcmc_samples)
        model = ModelListGP(gp1, gp2)
    else:
        model = SaasFullyBayesianSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=outcome_transform,
        )
        model.load_mcmc_samples(mcmc_samples)

    return model

class TestSelfCorrectingBayesianOptimization(BotorchTestCase):
        
    def test_scorebo(self):
        torch.manual_seed(1)
        tkwargs = {"device": self.device}
        estimation_types = ["LB"] # MC not implemented yet (the other option)
        distance_metrics = ("hellinger", "wasserstein", "kl_divergence")
        num_objectives = 1
        num_models = 3
        for (
            dtype,
            estimation_type,
            distance_metric,
            only_maxval,
            use_model_list,
            standardize_model,
            maximize,
            infer_noise,
        ) in product(
            (torch.float, torch.double),
            estimation_types,
            distance_metrics,
            (False, True), # only_maxval
            (False, ), # use_model_list
            (False, True), # standardize_model
            (False, True), # maximize
            (True, ), # infer_noise - only one option avail in PyroModels
        ):
            tkwargs["dtype"] = dtype
            input_dim = 2
            train_X = torch.rand(4, input_dim, **tkwargs)
            train_Y = torch.rand(4, num_objectives, **tkwargs)

            model = get_model(
                train_X,
                train_Y,
                num_models,
                use_model_list,
                standardize_model,
                infer_noise,
                **tkwargs,
            )

            num_optimal_samples = 5
            optimal_inputs = torch.rand(
                num_optimal_samples,
                num_models,
                input_dim,
                **tkwargs
            )

            # SCoreBO can work with only max-value, so we're testing that too
            if only_maxval:
                optimal_inputs = None
            optimal_outputs = torch.rand(
                num_optimal_samples,
                num_models,
                num_objectives,
                **tkwargs
            )

            # test acquisition
            X_pending_list = [None, torch.rand(2, input_dim, **tkwargs)]
            for i in range(len(X_pending_list)):
                X_pending = X_pending_list[i]

                acq = SelfCorrectingBayesianOptimization(
                    model=model,
                    optimal_inputs=optimal_inputs,
                    optimal_outputs=optimal_outputs,
                    distance_metric=distance_metric,
                    estimation_type=estimation_type,
                    X_pending=X_pending,
                    maximize=maximize,
                )
                self.assertIsInstance(acq.sampler, SobolQMCNormalSampler)

                test_Xs = [
                    torch.rand(4, 1, input_dim, **tkwargs),
                    torch.rand(4, 3, input_dim, **tkwargs),
                    torch.rand(4, 5, 1, input_dim, **tkwargs),
                    torch.rand(4, 5, 3, input_dim, **tkwargs),
                ]

                for j in range(len(test_Xs)):
                    acq_X = acq.forward(test_Xs[j])
                    acq_X = acq(test_Xs[j])
                    # assess shape
                    self.assertTrue(acq_X.shape == test_Xs[j].shape[:-2])

                    print(acq_X.shape)

        with self.assertRaises(ValueError):
            acq = SelfCorrectingBayesianOptimization(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                estimation_type="NOT_AN_ESTIMATION_TYPE",
                num_samples=64,
                X_pending=X_pending,
                maximize=maximize,
            )

        with self.assertRaises(ValueError):
            acq = SelfCorrectingBayesianOptimization(
                model=model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                distance_metric="NOT_A_DISTANCE",
                num_samples=64,
                X_pending=X_pending,
                maximize=maximize,
            )

        # Support with non-fully bayesian models is not possible. Thus, we
        # throw an error.
        non_fully_bayesian_model = SingleTaskGP(train_X, train_Y)
        with self.assertRaises(ValueError):
            acq = SelfCorrectingBayesianOptimization(
                model=non_fully_bayesian_model,
                optimal_inputs=optimal_inputs,
                optimal_outputs=optimal_outputs,
                estimation_type="LB",
            )
