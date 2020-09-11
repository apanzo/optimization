"""
Custom ANN definition.

This module contains the definition of an ANN comptable with the SMT Toolbox.
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
    ANN general class.

    Attributes:
        tbd (dict_keys): Settings to be declared.
        log_dir (str): Directory to store the ANN model.
        validation_points (dict): Validation points.
        optimized (bool): Whether the hyperparameters have been optimized.
    """

    def __getitem__(self,a):
        """
        Convenience method to acces the options dictionary directly.

        Args:
            a (str): List key.
        Returns:
            value (any): List value.
        """
        value = self.options[a]
        
        return value

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
        """
        # Set default values
        declare = self.options.declare
        for param in self.tbd:
            declare(param, None)

    def write_stats(self,dictionary_as_list,name):
        """
        Writes the statistics about the surrogate's training.

        Args:
            dictionary_as_list (list): Training statistics.
            name (str): Name of the file to be written.
        """
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
