from kerastuner.tuners.bayesian import BayesianOptimizationOracle
from .multi_execution_tuner_pytorch_cv import MultiExecutionTunerPyTorchCV

class BayesianPyTorchCV(MultiExecutionTunerPyTorchCV):
    """BayesianOptimization tuning with Gaussian process.

    """

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
        self.executions_per_trial = kwargs.pop("executions_per_trial") ########
        super().__init__(oracle=oracle,hypermodel=hypermodel,**kwargs)
