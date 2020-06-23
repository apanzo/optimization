"""
This is the visualization module.

visual - scatter 2D/3D, curve, surface tri (not quad)
"""

# Import pypi packages
##import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from pymoo.factory import get_visualization

# Import custom packages
from datamod import scale

def plot_raw(data,name,normalized=False,**kwargs):
    """
    Plot either a scatter, curve or surface plot.

    Arguments:
        data: data object
        name: visualization type
        normalized: whether the data is normalized

    Raises:
        ValueError: if the visualization method is not supported
        
    """
    
##    breakpoint()
    if normalized:
        data_all = np.concatenate((data.input,data.output),1)
    else:
        data_all = np.concatenate((data.coordinates,data.response),1)
    plot = get_visualization("scatter", angle=(30,-135))
    if name == "scatter":
        plot_type = None
    elif name == "surface":    
        plot_type = "other"
    else:
        raise ValueError("Visualization invalid / not yet supported")
    plot.add(data_all,plot_type=plot_type,**kwargs)
    plot.do()
##    plot.apply(lambda ax: ax.set_xlim([0,1]))
    plot.show()


def vis_design_space(res):
    """
    Visualize the design space in design coordinates.

    Arguments:
        res: results object
        
    """
    plot = get_visualization("scatter", title = "Design Space", axis_labels="x")
    plot.add(res.X, s=30, facecolors='none', edgecolors='r')
    plot.do()
##    plot.apply(lambda ax: ax.set_xlim(*res.problem.ranges[0][0]))
##    plot.apply(lambda ax: ax.set_ylim(*res.problem.ranges[0][1]))
##    plot.apply(lambda ax: ax.grid())
    plot.show()


def vis_objective_space(res):
    """
    Visualize the design space in objective coordinates.

    Arguments:
        res: results object
        
    """
    if res.F.shape[-1] <= 2:
        plot = get_visualization("scatter", title = "Objective Space", axis_labels="f")
        plot.add(res.F, s=30, facecolors='none', edgecolors='r')
    else:
        plot = get_visualization("pcp", title = "Objective Space", axis_labels="f")
        plot.add(res.F)
    plot.do()
##    plot.apply(lambda ax: ax.set_xlim(*res.problem.ranges[1][0]))
##    plot.apply(lambda ax: ax.set_ylim(*res.problem.ranges[0][1]))
##    plot.apply(lambda ax: ax.grid())
    plot.show()    

def show_problem(problem):
    """
    Show response of the defined problem.

    Arguments:
        problem: problem object
    """
    plot = get_visualization("fitness-landscape", problem, angle=(30,-135), _type="surface", kwargs_surface = dict(cmap="coolwarm", rstride=1, cstride=1))
##    plot.do()
##    breakpoint()
##    plot.apply(lambda ax: ax.set_zlim([0,1]))
    plot.show() 
##    get_visualization("fitness-landscape", problem, _type="contour", colorbar=True, kwargs_contour = dict(cmap="coolwarm", linestyles="solid", offset=-1)).show()


def compare():
    """
    Plot the comparison of raw data and surrogate response.

    Notes:
        NOT USED
    """
    plt.scatter(model.surrogate.train_in,model.surrogate.train_out)
    plt.plot(model.data.input,model.surrogate.predict_values(model.data.input),"k")
    plt.show()

def sample_size_convergence(metric,name):
    """
    Plot the sample size convergence.

    Arguments:
        model: model object
        
    """
    plt.plot(metric)
    plt.title(name)
##    plt.ylim(ymin=0)
    plt.show()

def plot_2d_sample():
    pass
        ##        import matplotlib.pyplot as plt
    ##        plt.scatter(model.data.input[:,0],model.data.input[:,1])
    ##        plt.show()
