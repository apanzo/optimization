"""
Bayesian Optimization tuning for TensorFlow with Gaussian process.
"""

from . import multi_execution_tuner_cv
from kerastuner.tuners.bayesian import BayesianOptimizationOracle

class BayesianOptimizationCV(multi_execution_tuner_cv.MultiExecutionTunerCV):

    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 num_initial_points=2,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 **kwargs):
        oracle = BayesianOptimizationOracle(
            objective=objective,
            max_trials=max_trials,
            num_initial_points=num_initial_points,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)
        super(BayesianOptimizationCV, self, ).__init__(oracle=oracle,
                                                     hypermodel=hypermodel,
                                                     **kwargs)
