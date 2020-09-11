"""
Function definitions.

This module contains function definitions.

Attributes:
    problems (dict): Classes of the custom defined problems.
"""
# Import pypi packages
import numpy as np
from pymoo.model.problem import Problem

class Custom(Problem):
    """
    Class for custom built problems using a surrogate.

    Attributes:
        function (): The response function.    
    """
    def __init__(self, function, xl, xu, n_obj, n_constr):
        """
        Args:
            function (): The response function.
            xl (np.array): Lower bounds of input coordinates.
            xu (np.array): Upper bounds of input coordinates.
            n_const (int): Number of constraints.
            n_obj (int): Number of objectives.
        """
        if len(xl) != len(xu):
            raise ValueError('Incorrent bounds')
        n_var = len(xl)
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, type_var=np.double)
        self.function = function
        
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.function(x)[:,:-self.n_constr or None]
        out["G"] = self.function(x)[:,-self.n_constr:]

class GettingStarted(Problem):
    """
    Pymoo example problem.

    References:
        http://pymoo.org/getting_started.html
    """

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:,0]**2 + x[:,1]**2
        f2 = (x[:,0]-1)**2 + x[:,1]**2

        g1 = 2*(x[:, 0]-0.1) * (x[:, 0]-0.9) / 0.18
        g2 = - 20*(x[:, 0]-0.4) * (x[:, 0]-0.6) / 4.8

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

class MatlabPeaks(Problem):
    """
    MATLAB peaks function definition.

    References:
        https://www.mathworks.com/help/matlab/ref/peaks.html
    """

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([-2,-3]),
                         xu=np.array([2,3]))

    def _evaluate(self, x, out, *args, **kwargs):
        x, y = x[:,0], x[:,1]
        f1 = 3*(1-x)**2*np.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) - 1/3*np.exp(-(x+1)**2 - y**2)

        out["F"] = f1

    def _calc_pareto_front(self, flatten=True, **kwargs):
        f = -6.55113332237566

        return f

    def _calc_pareto_set(self, flatten=True, **kwargs):
        x1 = 0.22826413206603302
        x2 = -1.625512756378189
        
        return [x1,x2]

class Squared(Problem):
    """
    Squared function.

    """

    def __init__(self):
        super().__init__(n_var=1,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([-2]),
                         xu=np.array([2]))

    def _evaluate(self, x, out, *args, **kwargs):
##        f1 = x**2
        f1 = (x-0.5)**2 - 0.3

        out["F"] = f1

class CubicSquared(Problem):
    """
    Squared function.

    """

    def __init__(self):
        super().__init__(n_var=1,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([-2]),
                         xu=np.array([2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = -x**2 + x**4 + 3

        out["F"] = f1
        

problems = {
    "matlab_peaks": MatlabPeaks,
    "squared": Squared,
    "cubic_squared": CubicSquared,
    "getting_started": GettingStarted}
