"""
This is the visualization module.

visual - scatter 2D/3D, curve, surface tri (not quad)
"""

# Import pypi packages
import numpy as np

from datamod.sampling import sample
from visumod.plots import scatter,scatter_pymoo,curve,heatmap,pcp,surface_pymoo

def plot_raw(data,iteration,normalized=False):
    """
    Plot either a scatter, curve or surface plot.

    Arguments:
        data: data object
        name: visualization type
        normalized: whether the data is normalized

    Raises:
        ValueError: if the visualization method is not supported

    Notes:
        * surface plot not used
    """
    # Select data to plot
    if normalized:
        data_all = np.concatenate((data.input,data.output),1)
    else:
        data_all = np.concatenate((data.coordinates,data.response),1)

##    # Select plot type
##    if name == "scatter":
##        plot_type = None
##    elif name == "surface":
##        plot_type = "other"

    # Save figure
    name = f"iteration_{iteration}_raw"
    scatter_pymoo(data_all,name+"_1","x")
        
def vis_design_space(data,iteration):
    """
    Visualize the design space in design coordinates.

    Arguments:
        res: results object
        
    """
    scatter_pymoo(data,f"optimization_{iteration}_design_space","x",s=30,facecolors='w',edgecolors='r')

def vis_objective_space(data,iteration):
    """
    Visualize the design space in objective coordinates.

    Arguments:
        res: results object
        
    """
    scatter_pymoo(data,f"optimization_{iteration}_objective_space_scatter","f",s=30,facecolors='w',edgecolors='k')

def vis_objective_space_pcp(data,iteration):
    """
    Visualize the design space in objective coordinates.

    Arguments:
        res: results object
        
    """
    pcp(data,f"optimization_{iteration}_objective_space_pcp")

def compare_surrogate(inputs,outputs,predict,iteration):
    """
    Plot the comparison of raw data and surrogate response.
    """
    data_all = np.stack((outputs.flatten(),predict(inputs).flatten()),1)
##    plt.plot([-1,1],[-1,1])
##    plt.xlabel("Training data")
##    plt.ylabel("Surrogate prediction")
    scatter(data_all,f"iteration_{iteration}_surrogate",compare=True)

def sample_size_convergence(metrics,name):
    """
    Plot the sample size convergence.

    Arguments:
        model: model object
        
    """
    curve(metrics,f"ssd_metric_{name}")

def correlation_heatmap(predict,ranges):
    x = sample("grid",1000,ranges.shape[0])
    y = predict(x)
    data = np.concatenate((x,y),1)
    cor = np.corrcoef(data,rowvar=False)
    heatmap(cor)

def surrogate_response(inputs,outputs,dimensions):
    data_all = np.concatenate((inputs,outputs),1)
    surface_pymoo(data_all)
