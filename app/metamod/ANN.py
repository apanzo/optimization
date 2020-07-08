"""
Custom ANN definition using TensorFlow.

This module contains the definition of an ANN comptable
with the SMT Toolbox
"""
# Import native packages
from collections import defaultdict
from tabulate import tabulate
import os

# Import pypi packages
from kerastuner.tuners import RandomSearch, BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters
import matplotlib.pyplot as plt
import numpy as np
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.checks import check_2d_array
from tensorflow import keras
from tensorflow_model_optimization.sparsity import keras as sparsity

# Import custom packages
from settings import load_json, settings
from visumod.plots import learning_curves
from .kerastuner.bayesian_custom import BayesianOptimizationCustom
        
class ANN(SurrogateModel):
    """
    ANN class.

    See also:
        Tensorflow documentation
        
    To do:
        * check if pretraining
        * if pretrainning:
            * select hyperparameters to optimize, keep rest default
            * take defaults from settings
            * select optimization method
            * perform pretraining
            * select best model
            * save best model
        * else:
            * load hyperparameters from optimization
    """

    def __getitem__(self,a):
        return self.options[a]

    def __init__(self,**kwargs):
        # To be declared
        self.tbd = kwargs.keys()
        super().__init__(**kwargs)

        # Initialize model
        self.name = "ANN"
        self.log_dir = os.path.join(settings["folder"],"logs","ann")

        self.validation_points = defaultdict(dict)

        self.optimized = False
        self._is_trained = False
        
    def _initialize(self):
        """
        Initialize the default hyperparameter settings.

        Parameters:
            no_layers: number of layers
            no_neurons: number of neurons per layer
            activation: activation functoin
            batch_size: batch size
            no_epochs: nu,ber of training epochs
            init: weight initialization strategy
            bias_init:  bias initialization strategy
            optimizer: optiimzer type
            loss: loss function
            kernel_regularizer: regularization paremeres
            dims: number of input and output dimension
            no_points: number of sample points
            prune: whether to use pruning
            sparsity: target network sparsity (fraction of zero weights)
            pruning_frequency: frequency of pruning
            tensorboard: whether to make tensorboard output
            stopping: use early stopping
            stopping_delta: required error delta threshold to stop training
            stopping_patience: number of iterations to wait before stopping
            plot_history: whether to plot the training history
        """
##        # Set default values
        declare = self.options.declare
        for param in self.tbd:
            declare(param, None)

    def build_hypermodel(self,hp):
        """
        General claass to build the ANN using tensorflow with autokeras hyperparameters defined

        Notes:
            * hyperparameters initialized using default values in config file
            * bias deactivated in output layer as it is centered around 0
        """
        # Initiallze model
        model = keras.Sequential()
        
        # Initialize hyperparameters        
        neurons_hyp = hp.Fixed("no_neurons",self["no_neurons"])
        layers_hyp = hp.Fixed("no_hid_layers",self["no_layers"])
        lr_hyp = hp.Fixed("learning_rate",self["learning_rate"])
        activation_hyp = hp.Fixed("activation_function", self["activation"])
        regularization_hyp = hp.Fixed("regularization",self["kernel_regularizer_param"])
        sparsity_hyp = hp.Fixed("sparsity",self["sparsity"])
        #

        # Load regularizer
        if self["kernel_regularizer"] == "l1":
            kernel_regularizer = keras.regularizers.l1(regularization_hyp)
        elif self["kernel_regularizer"] == "l2":
            kernel_regularizer = keras.regularizers.l2(regularization_hyp)
        else:
            raise Exception("Invalid regularized specified")
        
        # Add layers        
        in_dim, out_dim = self["dims"]
        model.add(keras.layers.Dense(neurons_hyp, activation=activation_hyp,
                                             kernel_initializer=self["init"],
                                             bias_initializer=self["bias_init"],
                                             kernel_regularizer=kernel_regularizer,
                                             input_shape=(in_dim,)))
        for _ in range(layers_hyp-1):
            model.add(keras.layers.Dense(neurons_hyp, activation=activation_hyp,
                                                 kernel_initializer=self["init"],
                                                 bias_initializer=self["bias_init"],
                                                 kernel_regularizer=kernel_regularizer))
        model.add(keras.layers.Dense(out_dim,activation="linear",use_bias=False))

        # Train
        optimizer = keras.optimizers.get(self["optimizer"])
        optimizer._hyper["learning_rate"] = lr_hyp

        # Prune the model
        if self["prune"]:
            model = self.prune_model(model,sparsity_hyp)

        model.compile(optimizer,self["loss"],metrics=["mse","mae","mape"])

        return model

    def prune_model(self,model,target_sparsity):
        """
        Notes:
        * Only constant sparsity active
        """
        if True:
            model = sparsity.prune_low_magnitude(model, sparsity.ConstantSparsity(target_sparsity, 0, frequency=10)        )
        else:
            end_step = np.ceil(self["no_points"]/self["batch_size"]*self["no_epochs"])
            print(f"End step: {end_step}")
            model = sparsity.prune_low_magnitude(model, sparsity.PolynomialSparsity(0,target_sparsity, 0, end_step, frequency=10)        )

        return model

    def get_callbacks(self):
        """
        Docstring

        Notes:
            * MyStopping never used
        """
        callbacks = []

        if self["prune"]:
            callbacks.append(sparsity.UpdatePruningStep())
        if self["tensorboard"]:
            raise ValueError("Tensorboard not implemented yet")
            callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(self.log_dir,"tensorboard"), histogram_freq=20))
        if self["stopping"]:
            if True:
                callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=self["stopping_delta"], patience=self["stopping_patience"], verbose=1,restore_best_weights=True))
            else:
                callbacks.append(keras.callbacks.MyStopping(monitor='val_loss', target=5, patience=stopping_patience, verbose=1,restore_best_weights=True))

        return callbacks

    def pretrain(self,inputs,outputs):
        print("### Performing Keras Tuner optimization of the ANN###")
        # Select hyperparameters to tune
        hp = HyperParameters()
        valid_entries = ["activation","neurons","layers","learning_rate","regularization","sparsity"]
        if not all([entry in valid_entries for entry in self["optimize"]]):
            raise Exception("Invalid hyperparameters specified for optimization")
        
        if "activation" in self["optimize"]:
            hp.Choice("activation_function",["sigmoid","relu","swish","tanh"],default=self["activation"])
        if "neurons" in self["optimize"]:
            hp.Int("no_neurons",6,20,step=2,default=self["no_neurons"])
        if "layers" in self["optimize"]:
            hp.Int("no_hid_layers",6,12,default=self["no_layers"])
        if "learning_rate" in self["optimize"]:
            hp.Float("learning_rate",0.001,0.1,sampling="log",default=self["learning_rate"])
        if "regularization" in self["optimize"]:
            hp.Float("regularization",0.0001,0.01,sampling="log",default=self["kernel_regularizer"])
        if "sparsity" in self["optimize"]:
            hp.Float("sparsity",0.3,0.9,default=self["sparsity"])

        no_hps = len(hp._space)
        
        # Load tuner
        max_trials = max(self["max_trials"]*no_hps,1)
        executions_per_trial = 1

        path_tf_format = "logs"
##        if self["tuner"] == "random":
##            tuner = RandomSearch(self.build_hypermodel,objective="val_mse",hyperparameters=hp,
##                                 max_trials=max_trials,executions_per_trial=1,directory="logs",overwrite=True)
##        elif self["tuner"] == "bayesian":
##            tuner = BayesianOptimization(self.build_hypermodel,objective="val_mae",hyperparameters=hp,
##                                 max_trials=max_trials,executions_per_trial=1,directory="logs",num_initial_points=3*len(hp._space),
##                                         overwrite=False,tune_new_entries=False,project_name="opt")
        tuner = BayesianOptimizationCustom(self.build_hypermodel,objective="val_mae",hyperparameters=hp,
                                 max_trials=max_trials,executions_per_trial=executions_per_trial,directory="logs",num_initial_points=3*no_hps,
                                         overwrite=True,tune_new_entries=False,project_name="opt")
        # Load data and callbacks
##        train_in, train_out = self.training_points[None][0]
##        test_in, test_out = self.validation_points[None][0]
        callbacks_all = self.get_callbacks()

        # Remove early stopping
        callbacks = [call for call in callbacks_all if not isinstance(call,keras.callbacks.EarlyStopping)]

        # Optimize
        tuner.search(inputs,outputs, 
            epochs=self["tuner_epochs"], verbose=0, shuffle=True, callbacks=callbacks)

        # Retrieve and save best model
        best_hp = tuner.get_best_hyperparameters()[0]
        untrained_model = tuner.hypermodel.build(best_hp)
        untrained_model.save(self.log_dir+"_untrained")

        scores = [tuner.oracle.trials[trial].score for trial in tuner.oracle.trials]
        hps = [tuner.oracle.trials[trial].hyperparameters.values for trial in tuner.oracle.trials]

        for idx,entry in enumerate(hps):
            entry["score"] = scores[idx]

        self.write_stats(hps,"ann_tuner_stats")

    def _train(self):
        """
        Train the ANN.

        API function: train the neural net
        """
        # Load untrained optimized model
        self.model = keras.models.load_model(self.log_dir+"_untrained")
        
        # Load data and callbacks
        train_in, train_out = self.training_points[None][0]
        test_in, test_out = self.validation_points[None][0]
        callbacks = self.get_callbacks()

        # Train the ANN
        histor = self.model.fit(train_in, train_out, batch_size = train_in.shape[0],
            epochs=self["no_epochs"], validation_data = (test_in, test_out), verbose=0, shuffle=True, callbacks=callbacks)

        # Evaluate the model
        self.validation_metric = self.evaluation(histor.history)
        self._is_trained = True
        if self.options["plot_history"]:
            self.plot_training_history(histor,train_in,train_out,self.model.predict)
       
    def evaluation(self,history):
        """
        Evaluate the generalization error.

        Returns:
            error: validation metric
        """
        mse = history["val_mean_squared_error"][-1]
        rmse = np.sqrt(mse)
        mae = history['val_mean_absolute_error'][-1]
        print(f'MSE: {mse:.3f}, RMSE: {rmse:.3f}')
        print(f"MAE: {mae:.3f}")
        stats = [{"MSE":mse,"RMSE":rmse,"MAE":mae}]
        self.write_stats(stats,"ann_training_stats")
##        print(f"MAPE: {history['val_mean_absolute_percentage_error'][-1]:.0f}")

        return mse

    def _predict_values(self, x):
        """
        API method: predict values using appropriate methods from the neural_network.py module

        :param  x: np.ndarray[n, nx] -- Input values for the prediction points
        :return y: np.ndarray[n, ny] -- Output values at the prediction points
        """
        return self.model.predict(x)


    def plot_training_history(self,history,inputs,outputs,predict):
        """
        Plot the evolution of the training and testing error.

        Arguments:
            history: training history object

        Raises:
            ValueError: if the surrogate has not been trained yet
            
        """
        if self._is_trained:
            learning_curves(history.history['loss'],history.history['val_loss'],outputs,predict(inputs),self.progress)
        else:
            raise Exception("Can't plot, the ANN is not trained yet")

    def write_stats(self,dictionary_as_list,name):
        path = os.path.join(settings["folder"],"logs",name+".txt")
        kwargs = {"headers":"keys"}
        if name == "ann_training_stats":
            kwargs.update({"floatfmt":".3f"})
            if self.progress[1] != 1:
                del kwargs["headers"]

        table = tabulate(dictionary_as_list,tablefmt="jira",**kwargs)
        
        if self.progress[1] == 1:
            with open(path, "a") as file:
                file.write("-------------------------")
                file.write("\n")
                file.write(f"       Iteration {self.progress[0]}      ")
                file.write("\n")
                file.write("-------------------------")
                file.write("\n")
        with open(path, "a") as file:
            file.write(table)
            file.write("\n")
##    def write_stats(self,mse,rmse,mae):
##        path = os.path.join(settings["folder"],"logs","ann_error_stats.txt")
##        if self.progress[0] == 1 and self.progress[1] == 1:
##            with open(path, "w") as file:
##                lines = tabulate([["MSE","RMSE","MAE"]],tablefmt="jira")
##                file.write(lines)
##                file.write("\n")
##        if self.progress[1] == 1:
##            with open(path, "a") as file:
##                file.write("-------------------------")
##                file.write("\n")
##                file.write(f"       Iteration {self.progress[0]}      ")
##                file.write("\n")
##                file.write("-------------------------")
##                file.write("\n")
##        with open(path, "a") as file:
##            lines = tabulate([[mse,rmse,mae]],tablefmt="jira", floatfmt=".3f")
##            file.write(lines)
##            file.write("\n")

    def set_validation_values(self, xt, yt, name=None):
        """
        Set validation data (values).

        Parameters
        ----------
        xt : np.ndarray[nt, nx] or np.ndarray[nt]
            The input values for the nt training points.
        yt : np.ndarray[nt, ny] or np.ndarray[nt]
            The output values for the nt training points.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        """
        xt = check_2d_array(xt, "xt")
        yt = check_2d_array(yt, "yt")

        if xt.shape[0] != yt.shape[0]:
            raise ValueError(
                "the first dimension of xt and yt must have the same length"
            )

        self.nt = xt.shape[0]
        self.nx = xt.shape[1]
        self.ny = yt.shape[1]
        kx = 0
        self.validation_points[name][kx] = [np.array(xt), np.array(yt)]
