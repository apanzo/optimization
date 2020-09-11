"""
Tuner for PyTorch that runs multiple executions per trial.

Notes:
    In fact, it does not.
"""
import collections
import copy

from kerastuner.engine import base_tuner
import numpy as np
import torch

from core.settings import settings
from metamod.validation import set_validation

class MultiExecutionTunerPyTorchCV(base_tuner.BaseTuner):

    def run_trial(self,
                  trial,
                  X,
                  y, *fit_args, **fit_kwargs):
        original_callbacks = fit_kwargs.pop('callbacks', [])
        
        metrics = collections.defaultdict(list)

        validation = settings["surrogate"]["validation"]
        validation_param = settings["surrogate"]["validation_param"]
        cv_split = set_validation(validation,validation_param)

        no_splits = cv_split.get_n_splits()
        iteration = fit_kwargs.pop("iteration_no")

        for idx, (train_indices, test_indices) in enumerate(cv_split.split(X)):
            X_train = torch.Tensor(X[train_indices])
            y_train = torch.Tensor(y[train_indices])
            X_test = torch.Tensor(X[test_indices])
            y_test = torch.Tensor(y[test_indices])

            metrics_avg = collections.defaultdict(list)
            for execution in range(self.executions_per_trial):
                copied_fit_kwargs = copy.copy(fit_kwargs)

                model = self.hypermodel.build(trial.hyperparameters)
                history = model.fit(fit_kwargs["epochs"],X_train,y_train,X_test,y_test,optimizing=True)

                for metric, epoch_values in history.history.items():
                    if self.oracle.objective.direction == 'min':
                        best_value = np.min(epoch_values)
                    else:
                        best_value = np.max(epoch_values)
                    metrics_avg[metric].append(best_value)

            # Average the results across executions and send to the Oracle.
            for metric, execution_values in metrics_avg.items():
                metrics[metric].append(np.mean(execution_values))

        trial_metrics = {name: np.mean(values) for name, values in metrics.items()}
        self.oracle.update_trial(trial.trial_id, trial_metrics)
