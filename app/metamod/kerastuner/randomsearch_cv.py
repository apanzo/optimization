"""
Random search tuning for TensorFlow.
"""

from . import multi_execution_tuner_cv
from kerastuner.tuners.randomsearch import RandomSearchOracle

class RandomSearchCV(multi_execution_tuner_cv.MultiExecutionTunerCV):

    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 **kwargs):
        self.seed = seed
        oracle = RandomSearchOracle(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)
        super(RandomSearchCV, self).__init__(
            oracle,
            hypermodel,
            **kwargs)
