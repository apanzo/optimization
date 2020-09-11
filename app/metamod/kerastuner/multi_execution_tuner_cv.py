"""
Tuner for TensorFlow that runs multiple executions per trial.
""""

from kerastuner.engine.multi_execution_tuner import MultiExecutionTuner
from kerastuner.engine import tuner_utils

import collections
import copy
import numpy as np

from core.settings import settings
from metamod.validation import set_validation

class MultiExecutionTunerCV(MultiExecutionTuner):

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
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]

            metrics_avg = collections.defaultdict(list)
            for execution in range(self.executions_per_trial):
                copied_fit_kwargs = copy.copy(fit_kwargs)
                callbacks = self._deepcopy_callbacks(original_callbacks)
                self._configure_tensorboard_dir(callbacks, trial.trial_id, execution)
                callbacks.append(tuner_utils.TunerCallback(self, trial))

                copied_fit_kwargs['callbacks'] = callbacks

                model = self.hypermodel.build(trial.hyperparameters)
                history = model.fit(X_train,y_train,*fit_args,validation_data = (X_test, y_test),
                                    batch_size = X_train.shape[0],**copied_fit_kwargs)

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

