"""
Custom ANN definition.

This module contains the definition of an ANN comptable
with the SMT Toolbox
"""
# Import native packages
from collections import defaultdict
import os
from tabulate import tabulate

# Import pypi packages
import numpy as np
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.checks import check_2d_array

# Import custom packages
from core.settings import settings
        
class ANN_base(SurrogateModel):
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

    def write_stats(self,dictionary_as_list,name):
        path = os.path.join(settings["folder"],"logs",name+".txt")
        kwargs = {"headers":"keys"}
        if name == "ann_training_stats":
            kwargs.update({"floatfmt":".3f"})

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
