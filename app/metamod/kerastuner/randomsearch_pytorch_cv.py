"""
Random search tuning for PyTorch.
"""

from kerastuner.tuners.randomsearch import RandomSearchOracle
from .multi_execution_tuner_pytorch_cv import MultiExecutionTunerPyTorchCV

class RandomSearchPyTorchCV(MultiExecutionTunerPyTorchCV):

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
        self.executions_per_trial = kwargs.pop("executions_per_trial") ########
        super().__init__(oracle,hypermodel,**kwargs)
