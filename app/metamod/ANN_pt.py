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
import matplotlib.pyplot as plt
import numpy as np
##import torch
##import torch.nn as nn
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.checks import check_2d_array

# Import custom packages
from core.settings import load_json, settings
from .kerastuner.bayesian_cv import BayesianOptimizationCV
from .kerastuner.randomsearch_cv import RandomSearchCV
from visumod import plot_training_history
        
class ANN_pt(SurrogateModel):
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
        self.name = "ANN_pt"
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
        # Set default values
        declare = self.options.declare
        for param in self.tbd:
            declare(param, None)

    def _train(self):
        """
        Train the ANN.

        API function: train the neural net
        """
        # Load untrained optimized model
        in_dim, out_dim = self["dims"]

        layers = []
        layers.append(nn.Linear(in_dim,self["no_neurons"]))
        layers.append(nn.ReLU())
        for _ in range(self["no_neurons"]-1):
            layers.append(nn.Linear(self["no_neurons"],self["no_neurons"]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self["no_neurons"],out_dim))

        self.model = nn.Sequential(*layers)
        
        # Load data and callbacks
        train_in, train_out = self.training_points[None][0]
        test_in, test_out = self.validation_points[None][0]

        train_in = torch.Tensor(train_in)
        train_out = torch.Tensor(train_out)
        test_in = torch.Tensor(test_in)
        test_out = torch.Tensor(test_out)

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self["learning_rate"])

        epochs = self["no_epochs"]

        for t in range(epochs):
            self.model.eval()
            y_pred = self.model(train_in)
            loss = loss_fn(y_pred,train_out)
            y_eval = self.model(test_in)
            loss_eval = loss_fn(y_eval,test_out)
##            print(t, loss.item(), loss_eval.item())

            self.model.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._is_trained = True
       
    def evaluation(self,history):
        """
        Evaluate the generalization error.

        Returns:
            mse: mean squared error
        """
        breakpoint()
        mse = history["val_mse"]
        rmse = history["val_root_mean_squared_error"]
        mae = history['val_mae']
        print(f'MSE: {mse:.3f}, RMSE: {rmse:.3f}')
        print(f"MAE: {mae:.3f}")
        stats = [{"MSE":mse,"RMSE":rmse,"MAE":mae}]
        self.write_stats(stats,"ann_training_stats")
        
        return mse

    def _predict_values(self, x):
        """
        API method: predict values using appropriate methods from the neural_network.py module

        :param  x: np.ndarray[n, nx] -- Input values for the prediction points
        :return y: np.ndarray[n, ny] -- Output values at the prediction points
        """
        return self.model.predict(x)

    def write_stats(self,dictionary_as_list,name):
        path = os.path.join(settings["folder"],"logs",name+".txt")
        kwargs = {"headers":"keys"}
        if name == "ann_training_stats":
            kwargs.update({"floatfmt":".3f"})
##            if self.progress[1] != 1:
##                del kwargs["headers"]

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
