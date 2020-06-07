"""
Function definitions.

This module contains function definitions
"""
import numpy as np
from pymoo.model.problem import Problem

class Custom(Problem):
    """
    Class for custom built problems using a surrogate.

    Note:
        data should be always normalized
    """
    def __init__(self, surrogate, xl, xu, n_constr):
        """Constructor.

        Arguments:
            surrogate:
            xl:
            xu:
            n_constr:

        Raises:
            ValueError:

        """
        if len(xl) != len(xu):
            raise ValueError('Incorrent bounds')
        n_var = len(xl)
        n_obj = surrogate.ny - n_constr
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, type_var=np.double)
        self.function = surrogate.predict_values
        self.n_constr = n_constr
        
        
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.function(x)[:,:-self.n_constr or None]
        if self.n_constr:
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


##    # --------------------------------------------------
##    # Pareto-front - not necessary but used for plotting
##    # --------------------------------------------------
##    def _calc_pareto_front(self, flatten=True, **kwargs):
##        f1_a = np.linspace(0.1**2, 0.4**2, 100)
##        f2_a = (np.sqrt(f1_a) - 1)**2
##
##        f1_b = np.linspace(0.6**2, 0.9**2, 100)
##        f2_b = (np.sqrt(f1_b) - 1)**2
##
##        a, b = np.column_stack([f1_a, f2_a]), np.column_stack([f1_b, f2_b])
##        return stack(a, b, flatten=flatten)
##
##    # --------------------------------------------------
##    # Pareto-set - not necessary but used for plotting
##    # --------------------------------------------------
##    def _calc_pareto_set(self, flatten=True, **kwargs):
##        x1_a = np.linspace(0.1, 0.4, 50)
##        x1_b = np.linspace(0.6, 0.9, 50)
##        x2 = np.zeros(50)
##
##        a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
##        return stack(a,b, flatten=flatten)

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
